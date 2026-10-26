"""Exercise the orchestrator graph with fake agents and a fake chat model, without network."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.clinical_guideline_agent.schema import FileCitation, GuidelineAnswer, GuidelineStreamUpdate
from src.drugs_agent import DrugAnswer, DrugMatch
from src.orchestrator import ClinicalDrugOrchestrator
from src.prompts.orchestrator_system_prompts import (
    ANSWER_CONSOLIDATION_SYSTEM_PROMPT,
    DRUG_CLASS_EXTRACTION_SYSTEM_PROMPT,
    QUESTION_ROUTING_SYSTEM_PROMPT,
)

GUIDELINE_ANSWER = GuidelineAnswer(
    response_id="resp_1",
    text="Start an ACE inhibitor or a thiazide diuretic.",
    citations=[FileCitation(file_id="file_1", filename="hypertension.pdf", index=45)],
)
MARKED_GUIDELINE_TEXT = "Start an ACE inhibitor or a thiazide diuretic.${}^{[1]}$"


def drug(name: str, drug_class: str = "ACE INHIBITORS") -> DrugMatch:
    return DrugMatch(
        drug_name=name,
        drug_class=drug_class,
        drug_sub_class=None,
        therapeutic_category=None,
        route_of_administration="oral",
        formulation="tablet",
        dosage_strengths=["10 mg"],
    )


class FakeChatModel:
    """Answer each orchestrator prompt with a canned reply and record what it was sent."""

    def __init__(self, *, route, drug_classes=(), consolidated=("Answer.",)):
        self.replies = {
            QUESTION_ROUTING_SYSTEM_PROMPT: {"route": route},
            DRUG_CLASS_EXTRACTION_SYSTEM_PROMPT: {"drug_classes": list(drug_classes)},
        }
        self.consolidated = consolidated
        self.calls: list[tuple[str, str, dict]] = []

    async def execute_chat_completion(self, messages, *, stream=False, **kwargs):
        system_prompt, user_content = messages[0]["content"], messages[1]["content"]
        self.calls.append((system_prompt, user_content, kwargs))
        if stream:
            assert system_prompt == ANSWER_CONSOLIDATION_SYSTEM_PROMPT
            return self._stream()
        return SimpleNamespace(
            finish_reason="stop", content=json.dumps(self.replies[system_prompt])
        )

    async def _stream(self):
        for text_chunk in self.consolidated:
            yield text_chunk

    def call_for(self, system_prompt):
        return next(call for call in self.calls if call[0] == system_prompt)


def guideline_agent() -> MagicMock:
    agent = MagicMock()
    agent.answer_guideline_question = AsyncMock(return_value=GUIDELINE_ANSWER)

    async def stream(question):
        yield GuidelineStreamUpdate(type="text", text="Start an ACE inhibitor ")
        yield GuidelineStreamUpdate(type="text", text="or a thiazide diuretic.")
        yield GuidelineStreamUpdate(type="completed", answer=GUIDELINE_ANSWER)

    agent.stream_guideline_answer_updates = MagicMock(side_effect=stream)
    return agent


def drug_agent(
    answers_by_class: dict[str, list[DrugMatch]],
    answers_by_query: dict[str, list[DrugMatch]] | None = None,
) -> MagicMock:
    agent = MagicMock()

    async def answer(drug_class):
        matches = answers_by_class.get(drug_class, [])
        return DrugAnswer(found=bool(matches), matches=matches)

    async def answer_query(drug_search_query):
        matches = (answers_by_query or {}).get(drug_search_query, [])
        return DrugAnswer(found=bool(matches), matches=matches)

    agent.answer_drug_class_query = AsyncMock(side_effect=answer)
    agent.answer_drug_search_query = AsyncMock(side_effect=answer_query)
    return agent


async def collect_with_answer(orchestrator, question="How should hypertension be treated?"):
    """Return the streamed text chunks and the final answer, checking it comes last, once."""
    updates = [
        update async for update in orchestrator.stream_clinical_drug_answer_updates(question)
    ]
    *text_updates, completed = updates
    assert all(update.type == "text" for update in text_updates)
    assert completed.type == "completed" and completed.answer is not None
    return [update.text for update in text_updates], completed.answer


async def collect(orchestrator, question="How should hypertension be treated?"):
    text_chunks, _ = await collect_with_answer(orchestrator, question)
    return text_chunks


class ClinicalOnlyRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_clinical_question_streams_the_guideline_answer(self) -> None:
        guidelines, drugs = guideline_agent(), drug_agent({})
        chat_model = FakeChatModel(route="clinical")

        chunks, answer = await collect_with_answer(
            ClinicalDrugOrchestrator(guidelines, drugs, chat_model)
        )

        self.assertEqual(chunks, ["Start an ACE inhibitor ", "or a thiazide diuretic."])
        self.assertEqual(answer.route, "clinical")
        # Streamed text has no markers; the completed answer places them at the cited offsets.
        self.assertEqual(answer.text, MARKED_GUIDELINE_TEXT)
        self.assertEqual(answer.sources, GUIDELINE_ANSWER.citations)
        self.assertIs(answer.guideline_answer, GUIDELINE_ANSWER)
        self.assertIsNone(answer.drug_answer)
        self.assertEqual(answer.drug_classes, [])
        guidelines.answer_guideline_question.assert_not_awaited()
        drugs.answer_drug_class_query.assert_not_awaited()
        self.assertEqual([call[0] for call in chat_model.calls], [QUESTION_ROUTING_SYSTEM_PROMPT])


class DrugOnlyRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_catalog_question_searches_drugs_without_asking_the_guidelines(self) -> None:
        question = "What strengths does lisinopril come in?"
        guidelines = guideline_agent()
        drugs = drug_agent({}, {question: [drug("Lisinopril")]})
        chat_model = FakeChatModel(route="drug")

        chunks, answer = await collect_with_answer(
            ClinicalDrugOrchestrator(guidelines, drugs, chat_model), question
        )

        # The guidelines are never consulted, so nothing is cited.
        guidelines.answer_guideline_question.assert_not_awaited()
        guidelines.stream_guideline_answer_updates.assert_not_called()
        self.assertIsNone(answer.guideline_answer)
        self.assertEqual(answer.sources, [])
        self.assertEqual(answer.route, "drug")

        # The question reaches the drug agent unchanged; it does its own classifying.
        drugs.answer_drug_search_query.assert_awaited_once_with(question)
        drugs.answer_drug_class_query.assert_not_awaited()

        text = "".join(chunks)
        self.assertIn("- **Lisinopril**", text)
        self.assertEqual(answer.text, text)
        self.assertEqual([match.drug_name for match in answer.drug_answer.matches], ["Lisinopril"])
        self.assertEqual(answer.drug_classes, [])

    async def test_only_the_router_asks_the_model(self) -> None:
        """No classes are extracted and no answer is consolidated, so one call is enough."""
        chat_model = FakeChatModel(route="drug")

        await collect(ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), chat_model))

        self.assertEqual([call[0] for call in chat_model.calls], [QUESTION_ROUTING_SYSTEM_PROMPT])

    async def test_a_catalog_question_with_no_match_says_so(self) -> None:
        chat_model = FakeChatModel(route="drug")

        chunks, answer = await collect_with_answer(
            ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), chat_model)
        )

        self.assertIn("No matching drugs were found.", "".join(chunks))
        self.assertFalse(answer.drug_answer.found)


class ClinicalAndDrugRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_guidelines_then_classes_then_drugs_then_one_streamed_answer(self) -> None:
        guidelines = guideline_agent()
        drugs = drug_agent(
            {
                "angiotensin-converting enzyme inhibitors": [drug("Lisinopril")],
                "thiazide diuretics": [drug("Hydrochlorothiazide", "THIAZIDES")],
            }
        )
        chat_model = FakeChatModel(
            route="clinical_and_drug",
            drug_classes=["angiotensin-converting enzyme inhibitors", "thiazide diuretics"],
            consolidated=("Use an ", "ACE inhibitor."),
        )

        chunks, answer = await collect_with_answer(
            ClinicalDrugOrchestrator(guidelines, drugs, chat_model)
        )

        guidelines.answer_guideline_question.assert_awaited_once()
        guidelines.stream_guideline_answer_updates.assert_not_called()
        self.assertEqual(
            [call.args[0] for call in drugs.answer_drug_class_query.await_args_list],
            ["angiotensin-converting enzyme inhibitors", "thiazide diuretics"],
        )
        text = "".join(chunks)
        self.assertTrue(text.startswith("Use an ACE inhibitor.\n\n**Available in the catalog**"))
        self.assertIn("Lisinopril", text)
        self.assertIn("Hydrochlorothiazide", text)
        self.assertNotIn("Guideline sources", text)
        self.assertEqual(answer.route, "clinical_and_drug")
        self.assertEqual(answer.text, text)
        self.assertIs(answer.guideline_answer, GUIDELINE_ANSWER)
        self.assertEqual(
            [match.drug_name for match in answer.drug_answer.matches],
            ["Lisinopril", "Hydrochlorothiazide"],
        )

    async def test_classes_are_extracted_from_the_guideline_answer(self) -> None:
        chat_model = FakeChatModel(route="clinical_and_drug", drug_classes=["x"])

        await collect(ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), chat_model))

        _, user_content, _ = chat_model.call_for(DRUG_CLASS_EXTRACTION_SYSTEM_PROMPT)
        self.assertEqual(json.loads(user_content)["guideline_answer"], GUIDELINE_ANSWER.text)

    async def test_the_consolidation_sees_the_guideline_text_and_the_drug_records(self) -> None:
        drugs = drug_agent({"angiotensin-converting enzyme inhibitors": [drug("Lisinopril")]})
        chat_model = FakeChatModel(
            route="clinical_and_drug",
            drug_classes=["angiotensin-converting enzyme inhibitors"],
        )

        await collect(ClinicalDrugOrchestrator(guideline_agent(), drugs, chat_model))

        _, user_content, kwargs = chat_model.call_for(ANSWER_CONSOLIDATION_SYSTEM_PROMPT)
        sent = json.loads(user_content)
        self.assertEqual(sent["guideline_answer"], MARKED_GUIDELINE_TEXT)
        self.assertEqual([record["drug_name"] for record in sent["drug_records"]], ["Lisinopril"])
        self.assertNotIn("drug_sub_class", sent["drug_records"][0])
        self.assertEqual(kwargs["temperature"], 0.01)

    async def test_the_drug_route_keeps_known_markers_and_drops_invented_ones(self) -> None:
        chat_model = FakeChatModel(
            route="clinical_and_drug",
            drug_classes=[],
            consolidated=("Use an ACE inhibitor.${}^{[1]}$ ", "Avoid in pregnancy.${}^{[2]}$"),
        )

        _, answer = await collect_with_answer(
            ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), chat_model)
        )

        self.assertTrue(
            answer.text.startswith("Use an ACE inhibitor.${}^{[1]}$ Avoid in pregnancy.")
        )
        self.assertNotIn("${}^{[2]}$", answer.text)
        self.assertEqual(answer.sources, GUIDELINE_ANSWER.citations)

    async def test_classes_are_extracted_from_the_unmarked_guideline_text(self) -> None:
        chat_model = FakeChatModel(route="clinical_and_drug", drug_classes=["x"])

        await collect(ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), chat_model))

        _, user_content, _ = chat_model.call_for(DRUG_CLASS_EXTRACTION_SYSTEM_PROMPT)
        self.assertNotIn("${}^{[1]}$", json.loads(user_content)["guideline_answer"])

    async def test_at_most_five_classes_are_searched(self) -> None:
        drugs = drug_agent({})
        chat_model = FakeChatModel(
            route="clinical_and_drug", drug_classes=["a", "b", "c", "d", "e", "f"]
        )

        await collect(ClinicalDrugOrchestrator(guideline_agent(), drugs, chat_model))

        self.assertEqual(drugs.answer_drug_class_query.await_count, 5)

    async def test_the_consolidation_is_told_which_classes_were_searched(self) -> None:
        """So it cannot say the catalog lacks a class that was never looked up."""
        chat_model = FakeChatModel(route="clinical_and_drug", drug_classes=["a", "b"])

        await collect(ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), chat_model))

        _, user_content, _ = chat_model.call_for(ANSWER_CONSOLIDATION_SYSTEM_PROMPT)
        self.assertEqual(json.loads(user_content)["searched_drug_classes"], ["a", "b"])

    async def test_a_drug_found_in_two_classes_appears_once(self) -> None:
        drugs = drug_agent({"a": [drug("Lisinopril")], "b": [drug("Lisinopril")]})
        chat_model = FakeChatModel(route="clinical_and_drug", drug_classes=["a", "b"])

        answer = "".join(
            await collect(ClinicalDrugOrchestrator(guideline_agent(), drugs, chat_model))
        )

        self.assertEqual(answer.count("- **Lisinopril**"), 1)

    async def test_no_recommended_class_means_no_drug_search_and_no_table(self) -> None:
        drugs = drug_agent({})
        chat_model = FakeChatModel(route="clinical_and_drug", drug_classes=[])

        answer = "".join(
            await collect(ClinicalDrugOrchestrator(guideline_agent(), drugs, chat_model))
        )

        drugs.answer_drug_class_query.assert_not_awaited()
        self.assertIn("No matching drugs were found.", answer)


class FailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_blank_question_is_rejected(self) -> None:
        orchestrator = ClinicalDrugOrchestrator(
            guideline_agent(), drug_agent({}), FakeChatModel(route="clinical")
        )

        with self.assertRaises(ValueError):
            await collect(orchestrator, "   ")

    async def test_a_truncated_routing_reply_is_an_error(self) -> None:
        chat_model = MagicMock()
        chat_model.execute_chat_completion = AsyncMock(
            return_value=SimpleNamespace(finish_reason="length", content="{")
        )
        orchestrator = ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), chat_model)

        with self.assertRaisesRegex(RuntimeError, "QuestionRoutingDecision ended early: length"):
            await collect(orchestrator)


if __name__ == "__main__":
    unittest.main()
