"""Exercise the orchestrator graph with fake agents and a fake chat model, without network."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError

from src.adapters.llm_provider.structured_model_response import ask_model_for_structured_json
from src.clinical_guideline_agent.schema import (
    FileCitation,
    GuidelineAnswer,
    GuidelineStreamUpdate,
)
from src.drugs_agent import DrugMatch
from src.memory import create_in_memory_checkpointer
from src.orchestrator import ClinicalDrugOrchestrator
from src.orchestrator.final_answer import build_final_answer
from src.orchestrator.nodes import finalize_answer, stream_guideline_answer
from src.orchestrator.schema import QuestionTriage
from src.orchestrator.triage import classify_question
from src.prompts.orchestrator_system_prompts import (
    QUESTION_TRIAGE_SYSTEM_PROMPT,
)

GUIDELINE_ANSWER = GuidelineAnswer(
    response_id="resp_1",
    text="Start an ACE inhibitor or a thiazide diuretic.",
    citations=[FileCitation(file_id="file_1", filename="hypertension.pdf", index=45)],
)
MARKED_GUIDELINE_TEXT = "Start an ACE inhibitor or a thiazide diuretic.${}^{[1]}$"
GUIDELINE_CHUNKS = ["Start an ACE inhibitor ", "or a thiazide diuretic."]


def drug(name: str, drug_class: str | None = "ACE INHIBITORS") -> DrugMatch:
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

    def __init__(
        self,
        *,
        kind="clinical",
        drug_search_terms=(),
    ):
        self.replies = {
            QUESTION_TRIAGE_SYSTEM_PROMPT: {
                "kind": kind,
                "drug_search_terms": list(drug_search_terms),
            },
        }
        self.calls: list[tuple[str, str, dict]] = []
        self.full_messages_by_call: list[list[dict]] = []  # everything each call was sent

    async def execute_chat_completion(self, messages, *, stream=False, **kwargs):
        # System prompt first, current question last; any chat history sits in between.
        system_prompt, user_content = messages[0]["content"], messages[-1]["content"]
        self.calls.append((system_prompt, user_content, kwargs))
        self.full_messages_by_call.append(messages)
        if stream:
            raise AssertionError("No stock-rendering LLM call is allowed")
        return SimpleNamespace(
            finish_reason="stop", content=json.dumps(self.replies[system_prompt])
        )

    def call_for(self, system_prompt):
        return next(call for call in self.calls if call[0] == system_prompt)

    def was_called_for(self, system_prompt):
        return any(call[0] == system_prompt for call in self.calls)


def guideline_agent() -> MagicMock:
    agent = MagicMock()
    agent.answer_guideline_question = AsyncMock(return_value=GUIDELINE_ANSWER)

    async def stream(question):
        for text_chunk in GUIDELINE_CHUNKS:
            yield GuidelineStreamUpdate(type="text", text=text_chunk)
        yield GuidelineStreamUpdate(type="completed", answer=GUIDELINE_ANSWER)

    agent.stream_guideline_answer_updates = MagicMock(side_effect=stream)
    return agent


def drug_agent(
    answers_by_class: dict[str, list[DrugMatch]],
    answers_by_term: dict[str, list[DrugMatch]] | None = None,
) -> MagicMock:
    agent = MagicMock()

    async def stream_answer(*, question, drug_search_terms):
        matches = []
        for term in drug_search_terms:
            for match in (answers_by_term or answers_by_class).get(term, []):
                if match.drug_name not in [item.drug_name for item in matches]:
                    matches.append(match)
        if not matches:
            yield "No matching records were found."
            return
        yield "Listed in the catalog: "
        yield ", ".join(match.drug_name for match in matches) + "."

    agent.stream_drug_answer_text = MagicMock(side_effect=stream_answer)
    return agent


async def collect_updates(
    orchestrator, question="How should hypertension be treated?", conversation_thread_id=None
):
    """Return the text chunks, the status lines, and the final answer, in that order.

    Also checks the "completed" update comes last and comes only once.
    """
    updates = [
        update
        async for update in orchestrator.stream_clinical_drug_answer_updates(
            question, conversation_thread_id
        )
    ]
    *streamed, completed = updates
    assert all(update.type in ("text", "status") for update in streamed)
    assert completed.type == "completed" and completed.answer is not None
    return (
        [update.text for update in streamed if update.type == "text"],
        [update.text for update in streamed if update.type == "status"],
        completed.answer,
    )


async def collect_with_answer(
    orchestrator, question="How should hypertension be treated?", conversation_thread_id=None
):
    """Return the streamed text chunks and the final answer, ignoring status lines."""
    text_chunks, _, answer = await collect_updates(orchestrator, question, conversation_thread_id)
    return text_chunks, answer


async def collect(
    orchestrator, question="How should hypertension be treated?", conversation_thread_id=None
):
    text_chunks, _ = await collect_with_answer(orchestrator, question, conversation_thread_id)
    return text_chunks


class ClinicalPathNoDrugsTests(unittest.IsolatedAsyncioTestCase):
    """A guideline answer that recommends no drug class gets no stock list."""

    async def test_the_guideline_answer_is_streamed_and_is_the_whole_answer(self) -> None:
        guidelines, drugs = guideline_agent(), drug_agent({})
        chat_model = FakeChatModel(kind="clinical")

        chunks, answer = await collect_with_answer(
            ClinicalDrugOrchestrator(guidelines, drugs, chat_model)
        )

        self.assertEqual(chunks, GUIDELINE_CHUNKS)
        self.assertEqual(answer.route, "clinical")
        # The streamed text carries no markers; the finished answer places them at
        # the offsets the file search cited.
        self.assertEqual(answer.text, MARKED_GUIDELINE_TEXT)
        self.assertEqual(answer.sources, GUIDELINE_ANSWER.citations)
        self.assertEqual(answer.guideline_answer, GUIDELINE_ANSWER)

    async def test_the_catalog_is_never_searched_and_no_section_is_written(self) -> None:
        drugs = drug_agent({})
        chat_model = FakeChatModel(kind="clinical")

        await collect(ClinicalDrugOrchestrator(guideline_agent(), drugs, chat_model))

        self.assertEqual(len(chat_model.calls), 1)

    async def test_clinical_answer_does_not_trigger_another_model_call(self) -> None:
        model = FakeChatModel(kind="clinical")
        await collect(ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), model))
        self.assertEqual([prompt for prompt, _, _ in model.calls], [QUESTION_TRIAGE_SYSTEM_PROMPT])


class ClinicalPathWithDrugsTests(unittest.IsolatedAsyncioTestCase):
    """An explicit request for clinical advice and stock gets both agent outputs."""

    def chat_model(self, **overrides):
        return FakeChatModel(
            kind="clinical_and_drug",
            **overrides,
            drug_search_terms=["angiotensin-converting enzyme inhibitors"],
        )

    async def test_the_guidelines_stream_first_and_drug_agent_chunks_follow(self) -> None:
        model = self.chat_model()
        drugs = drug_agent({"angiotensin-converting enzyme inhibitors": [drug("Lisinopril")]})
        chunks, answer = await collect_with_answer(
            ClinicalDrugOrchestrator(guideline_agent(), drugs, model)
        )
        self.assertEqual(chunks[: len(GUIDELINE_CHUNKS)], GUIDELINE_CHUNKS)
        self.assertEqual(
            chunks[len(GUIDELINE_CHUNKS) :], ["\n\nListed in the catalog: ", "Lisinopril."]
        )
        self.assertEqual(
            answer.text, MARKED_GUIDELINE_TEXT + "".join(chunks[len(GUIDELINE_CHUNKS) :])
        )
        self.assertEqual(answer.sources, GUIDELINE_ANSWER.citations)
        self.assertEqual(answer.route, "clinical_and_drug")
        self.assertEqual([prompt for prompt, _, _ in model.calls], [QUESTION_TRIAGE_SYSTEM_PROMPT])

    async def test_the_guidelines_are_streamed_not_fetched_whole(self) -> None:
        guidelines = guideline_agent()

        await collect(ClinicalDrugOrchestrator(guidelines, drug_agent({}), self.chat_model()))

        guidelines.stream_guideline_answer_updates.assert_called_once()
        guidelines.answer_guideline_question.assert_not_awaited()

    async def test_explicit_queries_are_searched_in_the_order_requested(self) -> None:
        drugs = drug_agent({})
        chat_model = FakeChatModel(
            kind="clinical_and_drug", drug_search_terms=["first class", "second class"]
        )

        await collect(ClinicalDrugOrchestrator(guideline_agent(), drugs, chat_model))

        self.assertEqual(
            drugs.stream_drug_answer_text.call_args.kwargs["drug_search_terms"],
            ["first class", "second class"],
        )

    async def test_explicit_search_queries_are_not_limited_by_the_removed_class_cap(self) -> None:
        terms = ["a", "b", "c", "d", "e", "f"]
        drugs = drug_agent({})
        model = FakeChatModel(kind="clinical_and_drug", drug_search_terms=terms)
        await collect(ClinicalDrugOrchestrator(guideline_agent(), drugs, model))
        self.assertEqual(drugs.stream_drug_answer_text.call_args.kwargs["drug_search_terms"], terms)

    async def test_a_drug_found_under_two_queries_is_listed_once(self) -> None:
        drugs = drug_agent({"a": [drug("Lisinopril")], "b": [drug("Lisinopril")]})
        chat_model = FakeChatModel(kind="clinical_and_drug", drug_search_terms=["a", "b"])

        _, answer = await collect_with_answer(
            ClinicalDrugOrchestrator(guideline_agent(), drugs, chat_model)
        )

        self.assertEqual(answer.text.count("Lisinopril"), 1)

    async def test_empty_result_answer_comes_from_drug_agent(self) -> None:
        model = FakeChatModel(kind="clinical_and_drug", drug_search_terms=["a", "b"])
        _, answer = await collect_with_answer(
            ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), model)
        )
        self.assertTrue(answer.text.endswith("No matching records were found."))
        self.assertEqual(len(model.calls), 1)

    async def test_agent_text_is_preserved_without_python_display_limits(self) -> None:
        drugs = drug_agent({"a": [drug(f"Drug {number}") for number in range(7)]})
        model = FakeChatModel(kind="clinical_and_drug", drug_search_terms=["a"])
        _, answer = await collect_with_answer(
            ClinicalDrugOrchestrator(guideline_agent(), drugs, model)
        )
        for number in range(7):
            self.assertIn(f"Drug {number}", answer.text)


class CatalogOnlyPathTests(unittest.IsolatedAsyncioTestCase):
    """A question about what the pharmacy holds never reaches the guidelines."""

    async def test_class_abbreviation_is_passed_to_the_drug_agent_unchanged(self) -> None:
        drugs = drug_agent({})
        model = FakeChatModel(kind="catalog_only", drug_search_terms=["SSRIs"])

        await collect(
            ClinicalDrugOrchestrator(guideline_agent(), drugs, model), "List SSRIs in stock"
        )

        self.assertEqual(
            drugs.stream_drug_answer_text.call_args.kwargs["drug_search_terms"], ["SSRIs"]
        )

    async def test_a_catalog_question_skips_the_guidelines_entirely(self) -> None:
        guidelines = guideline_agent()
        drugs = drug_agent({}, {"sertraline": [drug("Sertraline", "SSRIS")]})
        chat_model = FakeChatModel(kind="catalog_only", drug_search_terms=["sertraline"])

        chunks, answer = await collect_with_answer(
            ClinicalDrugOrchestrator(guidelines, drugs, chat_model),
            "what strengths does sertraline come in",
        )

        guidelines.stream_guideline_answer_updates.assert_not_called()
        guidelines.answer_guideline_question.assert_not_awaited()
        self.assertEqual(
            [prompt for prompt, _, _ in chat_model.calls],
            [QUESTION_TRIAGE_SYSTEM_PROMPT],
        )

        # Nothing was cited, and the stock list starts the answer rather than following it.
        self.assertEqual(answer.route, "drug")
        self.assertIsNone(answer.guideline_answer)
        self.assertEqual(answer.sources, [])
        self.assertEqual("".join(chunks), answer.text)
        self.assertEqual(len(chunks), 2)
        self.assertIn("Sertraline", answer.text)

    async def test_the_catalog_is_searched_with_the_terms_triage_resolved(self) -> None:
        """A follow-up like "and its strengths?" is searched by the drug it refers to."""
        drugs = drug_agent({}, {})
        chat_model = FakeChatModel(kind="catalog_only", drug_search_terms=["sertraline"])

        await collect(
            ClinicalDrugOrchestrator(guideline_agent(), drugs, chat_model), "and its strengths?"
        )

        self.assertEqual(
            drugs.stream_drug_answer_text.call_args.kwargs["drug_search_terms"], ["sertraline"]
        )

    async def test_several_terms_are_searched_at_once(self) -> None:
        drugs = drug_agent({}, {})
        chat_model = FakeChatModel(
            kind="catalog_only", drug_search_terms=["sertraline", "fluoxetine"]
        )

        await collect(ClinicalDrugOrchestrator(guideline_agent(), drugs, chat_model))

        self.assertEqual(
            drugs.stream_drug_answer_text.call_args.kwargs["drug_search_terms"],
            ["sertraline", "fluoxetine"],
        )

    async def test_no_usable_term_falls_back_to_the_question_itself(self) -> None:
        drugs = drug_agent({}, {})
        chat_model = FakeChatModel(kind="catalog_only", drug_search_terms=["  "])

        await collect(ClinicalDrugOrchestrator(guideline_agent(), drugs, chat_model), "metformin?")

        self.assertEqual(
            drugs.stream_drug_answer_text.call_args.kwargs["drug_search_terms"], ["metformin?"]
        )


class StatusMessageTests(unittest.IsolatedAsyncioTestCase):
    """Say what is being worked on, so the clinician is not left watching a spinner."""

    async def test_the_clinical_path_reports_both_steps_in_order(self) -> None:
        drugs = drug_agent({"a": [drug("Lisinopril")]})
        chat_model = FakeChatModel(kind="clinical_and_drug", drug_search_terms=["a"])

        _, statuses, _ = await collect_updates(
            ClinicalDrugOrchestrator(guideline_agent(), drugs, chat_model)
        )

        self.assertEqual(
            statuses,
            ["Searching the clinical guidelines...", "Searching the catalog for a..."],
        )

    async def test_search_status_names_the_requested_queries(self) -> None:
        chat_model = FakeChatModel(kind="clinical_and_drug", drug_search_terms=["a", "b", "c"])

        _, statuses, _ = await collect_updates(
            ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), chat_model)
        )

        self.assertIn("Searching the catalog for a, b, c...", statuses)

    async def test_an_answer_with_no_drugs_reports_only_the_guideline_search(self) -> None:
        chat_model = FakeChatModel(kind="clinical")

        _, statuses, _ = await collect_updates(
            ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), chat_model)
        )

        self.assertEqual(statuses, ["Searching the clinical guidelines..."])

    async def test_the_catalog_only_path_names_what_it_is_searching_for(self) -> None:
        chat_model = FakeChatModel(kind="catalog_only", drug_search_terms=["sertraline"])

        _, statuses, _ = await collect_updates(
            ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}, {}), chat_model)
        )

        # The guidelines are never searched, so that status never appears.
        self.assertEqual(statuses, ["Searching the catalog for sertraline..."])

    async def test_a_status_line_never_becomes_part_of_the_answer(self) -> None:
        drugs = drug_agent({"a": [drug("Lisinopril")]})
        chat_model = FakeChatModel(kind="clinical_and_drug", drug_search_terms=["a"])

        chunks, statuses, answer = await collect_updates(
            ClinicalDrugOrchestrator(guideline_agent(), drugs, chat_model)
        )

        self.assertTrue(statuses)  # they were sent
        for status in statuses:
            self.assertNotIn(status, answer.text)
            self.assertNotIn(status, "".join(chunks))


class ConversationHistoryTests(unittest.IsolatedAsyncioTestCase):
    """Questions on the same conversation thread see earlier questions and answers."""

    async def ask_in_order(self, orchestrator, questions, conversation_thread_id):
        for question in questions:
            await collect_with_answer(orchestrator, question, conversation_thread_id)

    async def test_the_first_question_is_sent_with_no_history(self) -> None:
        chat_model = FakeChatModel(kind="clinical")

        await collect(ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), chat_model))

        triage_messages = chat_model.full_messages_by_call[0]
        self.assertEqual([message["role"] for message in triage_messages], ["system", "user"])

    async def test_omitting_thread_id_starts_a_fresh_history_on_each_call(self) -> None:
        model = FakeChatModel()
        orchestrator = ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), model)

        await collect(orchestrator, "first independent question")
        await collect(orchestrator, "second independent question")

        self.assertEqual(
            model.full_messages_by_call[1],
            [
                {"role": "system", "content": QUESTION_TRIAGE_SYSTEM_PROMPT},
                {"role": "user", "content": "second independent question"},
            ],
        )

    async def test_supplied_checkpointer_restores_history_in_another_orchestrator(self) -> None:
        checkpointer = create_in_memory_checkpointer()
        first = ClinicalDrugOrchestrator(
            guideline_agent(),
            drug_agent({"class A": [drug("Lisinopril")]}),
            FakeChatModel(drug_search_terms=["class A"], kind="clinical_and_drug"),
            checkpointer=checkpointer,
        )
        _, first_answer = await collect_with_answer(first, "first question", "shared-thread")

        model = FakeChatModel()
        second = ClinicalDrugOrchestrator(
            guideline_agent(),
            drug_agent({}),
            model,
            chat_completion_model_name="explicit-model",
            checkpointer=checkpointer,
        )
        saved = await second.graph.aget_state({"configurable": {"thread_id": "shared-thread"}})
        self.assertIsInstance(saved.values["guideline_answer"], GuidelineAnswer)
        self.assertNotIn("drug_answer", saved.values)
        self.assertEqual(saved.values["catalog_section_text"], "Listed in the catalog: Lisinopril.")
        self.assertIsInstance(saved.values["sources"][0], FileCitation)

        _, second_answer = await collect_with_answer(second, "second question", "shared-thread")

        self.assertIs(second.graph.checkpointer, checkpointer)
        self.assertEqual(
            model.full_messages_by_call[0][1:],
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": first_answer.text},
                {"role": "user", "content": "second question"},
            ],
        )
        self.assertTrue(all(kwargs["model"] == "explicit-model" for _, _, kwargs in model.calls))
        self.assertEqual(second_answer.text, MARKED_GUIDELINE_TEXT)
        self.assertEqual(second_answer.sources, GUIDELINE_ANSWER.citations)
        self.assertEqual(second_answer.route, "clinical")

    async def test_a_second_question_is_sent_with_the_first_question_and_answer(self) -> None:
        chat_model = FakeChatModel(kind="clinical")
        orchestrator = ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), chat_model)

        await self.ask_in_order(orchestrator, ["first question", "second question"], "thread-1")

        second_triage_messages = chat_model.full_messages_by_call[1]
        self.assertEqual(
            [(message["role"], message["content"]) for message in second_triage_messages[1:]],
            [
                ("user", "first question"),
                ("assistant", MARKED_GUIDELINE_TEXT),
                ("user", "second question"),
            ],
        )

    async def test_only_the_last_ten_exchanges_are_sent_with_the_current_question(self) -> None:
        chat_model = FakeChatModel(kind="clinical")
        orchestrator = ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), chat_model)
        twelve_questions = [f"question {number}" for number in range(1, 13)]

        await self.ask_in_order(orchestrator, twelve_questions, "thread-1")

        # The 12th question is sent with 10 earlier exchanges (20 messages), so
        # questions 1 and 2 have fallen out of the window.
        twelfth_triage_messages = chat_model.full_messages_by_call[-1]
        history_messages = twelfth_triage_messages[1:-1]
        self.assertEqual(len(history_messages), 20)
        self.assertEqual(history_messages[0]["content"], "question 2")
        self.assertEqual(history_messages[-2]["content"], "question 11")
        self.assertEqual(twelfth_triage_messages[-1]["content"], "question 12")

    async def test_different_threads_do_not_share_history(self) -> None:
        chat_model = FakeChatModel(kind="clinical")
        orchestrator = ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), chat_model)

        await collect(orchestrator, "question on thread one", "thread-1")
        await collect(orchestrator, "question on thread two", "thread-2")

        messages_for_thread_two = chat_model.full_messages_by_call[1]
        self.assertEqual(
            [message["role"] for message in messages_for_thread_two], ["system", "user"]
        )

    async def test_results_from_an_earlier_question_do_not_leak_into_the_next(self) -> None:
        drugs = drug_agent({"a": [drug("Lisinopril")]})
        chat_model = FakeChatModel(kind="clinical_and_drug", drug_search_terms=["a"])
        orchestrator = ClinicalDrugOrchestrator(guideline_agent(), drugs, chat_model)
        await collect(orchestrator, "drug question", "thread-1")

        chat_model.replies[QUESTION_TRIAGE_SYSTEM_PROMPT] = {
            "kind": "clinical",
            "drug_search_terms": [],
        }
        _, second_answer = await collect_with_answer(orchestrator, "guideline question", "thread-1")

        # The earlier question's drugs must not appear in a guideline-only answer.
        self.assertEqual(second_answer.text, MARKED_GUIDELINE_TEXT)


class FailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_or_invalid_triage_content_does_not_complete_an_answer(self) -> None:
        replies = (
            (None, RuntimeError, "QuestionTriage had no text"),
            ("not JSON", ValidationError, "Invalid JSON"),
            ('{"kind": "unknown", "drug_search_terms": []}', ValidationError, "kind"),
        )
        for content, error_type, message in replies:
            with self.subTest(content=content):
                model = MagicMock()
                model.execute_chat_completion = AsyncMock(
                    return_value=SimpleNamespace(finish_reason="stop", content=content)
                )
                guidelines, drugs = guideline_agent(), drug_agent({})
                orchestrator = ClinicalDrugOrchestrator(guidelines, drugs, model)
                updates = []

                with self.assertRaisesRegex(error_type, message):
                    async for update in orchestrator.stream_clinical_drug_answer_updates(
                        "question"
                    ):
                        updates.append(update)

                self.assertEqual(updates, [])
                guidelines.stream_guideline_answer_updates.assert_not_called()

                drugs.stream_drug_answer_text.assert_not_called()

    async def test_catalog_failure_propagates_without_rendering_or_completion(self) -> None:
        for kind in ("clinical_and_drug", "catalog_only"):
            with self.subTest(kind=kind):
                model = FakeChatModel(kind=kind, drug_search_terms=["Drug A"])
                drugs = drug_agent({})
                failure = RuntimeError("catalog unavailable")

                drugs.stream_drug_answer_text.side_effect = failure
                orchestrator = ClinicalDrugOrchestrator(guideline_agent(), drugs, model)
                updates = []

                with self.assertRaisesRegex(RuntimeError, "catalog unavailable"):
                    async for update in orchestrator.stream_clinical_drug_answer_updates(
                        "question"
                    ):
                        updates.append(update)

                self.assertFalse(any(update.type == "completed" for update in updates))
                self.assertEqual(len(model.calls), 1)
                text = [update.text for update in updates if update.type == "text"]
                self.assertEqual(text, GUIDELINE_CHUNKS if kind == "clinical_and_drug" else [])

    async def test_question_after_failed_search_keeps_history_without_stale_results(self) -> None:
        model = FakeChatModel(drug_search_terms=["class A"], kind="clinical_and_drug")
        drugs = drug_agent({})
        drugs.stream_drug_answer_text.side_effect = RuntimeError("catalog unavailable")
        orchestrator = ClinicalDrugOrchestrator(guideline_agent(), drugs, model)

        with self.assertRaisesRegex(RuntimeError, "catalog unavailable"):
            await collect(orchestrator, "failed question", "same-thread")

        model.replies[QUESTION_TRIAGE_SYSTEM_PROMPT] = {"kind": "clinical", "drug_search_terms": []}
        _, answer = await collect_with_answer(orchestrator, "next question", "same-thread")

        # A failed turn keeps its question, but has no finalized assistant answer.
        self.assertEqual(
            model.full_messages_by_call[1][1:],
            [
                {"role": "user", "content": "failed question"},
                {"role": "user", "content": "next question"},
            ],
        )
        self.assertEqual(answer.route, "clinical")
        self.assertEqual(answer.text, MARKED_GUIDELINE_TEXT)
        self.assertEqual(drugs.stream_drug_answer_text.call_count, 1)
        # One triage call for the failed turn and one for the next question.
        self.assertEqual(len(model.calls), 2)

    async def test_a_blank_question_is_rejected(self) -> None:
        orchestrator = ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), FakeChatModel())

        with self.assertRaises(ValueError):
            await collect(orchestrator, "   ")

    async def test_a_truncated_triage_reply_is_an_error(self) -> None:
        chat_model = MagicMock()
        chat_model.execute_chat_completion = AsyncMock(
            return_value=SimpleNamespace(finish_reason="length", content="{")
        )
        orchestrator = ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), chat_model)

        with self.assertRaisesRegex(RuntimeError, "QuestionTriage ended early: length"):
            await collect(orchestrator)

    async def test_a_guideline_stream_that_never_completes_is_an_error(self) -> None:
        guidelines = MagicMock()

        async def stream_without_completion(question):
            yield GuidelineStreamUpdate(type="text", text="partial")

        guidelines.stream_guideline_answer_updates = MagicMock(
            side_effect=stream_without_completion
        )
        orchestrator = ClinicalDrugOrchestrator(guidelines, drug_agent({}), FakeChatModel())

        with self.assertRaisesRegex(RuntimeError, "without a completed answer"):
            await collect(orchestrator)


class StreamingCallbackTests(unittest.IsolatedAsyncioTestCase):
    """Guideline nodes and catalog callbacks preserve immediate forwarding and failures."""

    async def test_guideline_chunks_are_forwarded_before_requesting_the_next_update(self) -> None:
        emitted = []

        async def stream(question):
            self.assertEqual(question, "question")
            yield GuidelineStreamUpdate(type="text", text="first")
            self.assertEqual(emitted, ["first"])
            yield GuidelineStreamUpdate(type="text", text="")
            self.assertEqual(emitted, ["first", ""])
            yield GuidelineStreamUpdate(type="completed", answer=GUIDELINE_ANSWER)
            self.fail("The completed answer must end forwarding")

        agent = SimpleNamespace(stream_guideline_answer_updates=stream)
        with (
            patch(
                "src.orchestrator.nodes.node_02_guideline_answer.send_answer_text",
                side_effect=emitted.append,
            ),
            patch("src.orchestrator.nodes.node_02_guideline_answer.send_status") as status,
        ):
            node_update = await stream_guideline_answer(
                {"clinical_question": "question"},
                SimpleNamespace(context=SimpleNamespace(guideline_agent=agent)),
            )

        self.assertIs(node_update["guideline_answer"], GUIDELINE_ANSWER)
        self.assertEqual(emitted, ["first", ""])
        status.assert_called_once_with("Searching the clinical guidelines...")

    async def test_incomplete_or_failed_stream_preserves_partial_text_and_error(self) -> None:
        for fails in (False, True):
            with self.subTest(fails=fails):
                emitted = []
                failure = RuntimeError("source stream failed")

                async def stream(question, should_fail=fails, stream_error=failure):
                    yield GuidelineStreamUpdate(type="text", text="partial")
                    if should_fail:
                        raise stream_error

                agent = SimpleNamespace(stream_guideline_answer_updates=stream)
                with (
                    patch(
                        "src.orchestrator.nodes.node_02_guideline_answer.send_answer_text",
                        side_effect=emitted.append,
                    ),
                    patch("src.orchestrator.nodes.node_02_guideline_answer.send_status"),
                    self.assertRaises(RuntimeError) as raised,
                ):
                    await stream_guideline_answer(
                        {"clinical_question": "question"},
                        SimpleNamespace(context=SimpleNamespace(guideline_agent=agent)),
                    )

                self.assertEqual(emitted, ["partial"])
                if fails:
                    self.assertIs(raised.exception, failure)
                else:
                    self.assertEqual(
                        str(raised.exception), "Guideline stream ended without a completed answer"
                    )


class StructuredModelReplyTests(unittest.IsolatedAsyncioTestCase):
    """The shared request helper returns the requested model and preserves failure rules."""

    async def test_success_returns_an_instance_of_the_requested_model(self) -> None:
        cases = (
            (QuestionTriage, {"kind": "clinical", "drug_search_terms": []}),
            (QuestionTriage, {"kind": "clinical_and_drug", "drug_search_terms": ["lisinopril"]}),
        )
        for output_model, payload in cases:
            with self.subTest(output_model=output_model.__name__):
                provider = MagicMock()
                provider.execute_chat_completion = AsyncMock(
                    return_value=SimpleNamespace(finish_reason="stop", content=json.dumps(payload))
                )

                reply = await ask_model_for_structured_json(
                    provider, "test-model", "instructions", "question", output_model
                )

                self.assertIsInstance(reply, output_model)
                self.assertEqual(reply.model_dump(), payload)
                provider.execute_chat_completion.assert_awaited_once()

    async def test_invalid_replies_keep_error_types_and_do_not_retry(self) -> None:
        cases = (
            ("length", "{", RuntimeError, "QuestionTriage ended early: length"),
            ("content_filter", None, RuntimeError, "QuestionTriage ended early: content_filter"),
            (
                "stop",
                None,
                RuntimeError,
                "QuestionTriage had no text; the model may have refused",
            ),
            ("stop", "", RuntimeError, "QuestionTriage had no text; the model may have refused"),
            ("stop", "not JSON", ValidationError, None),
            (
                "stop",
                '{"kind": "clinical", "drug_search_terms": "not a list"}',
                ValidationError,
                None,
            ),
        )
        for finish_reason, content, error_type, message in cases:
            with self.subTest(finish_reason=finish_reason, content=content):
                provider = MagicMock()
                provider.execute_chat_completion = AsyncMock(
                    return_value=SimpleNamespace(finish_reason=finish_reason, content=content)
                )

                with self.assertRaises(error_type) as raised:
                    await ask_model_for_structured_json(
                        provider, "test-model", "instructions", "question", QuestionTriage
                    )

                if message is not None:
                    self.assertEqual(str(raised.exception), message)
                provider.execute_chat_completion.assert_awaited_once()


class ModelDecisionBoundaryTests(unittest.IsolatedAsyncioTestCase):
    """Model decisions accept explicit inputs; nodes retain normalization and routing."""

    async def test_triage_sends_selected_history_and_returns_unmodified_terms(self) -> None:
        history = [
            {"role": "user", "content": "Drug A?"},
            {"role": "assistant", "content": "Earlier answer"},
        ]
        model = FakeChatModel(kind="catalog_only", drug_search_terms=[" Drug A ", " "])

        decision = await classify_question(
            question="and its strengths?",
            previous_messages=history,
            chat_completions_provider=model,
            model_name="test-model",
        )

        self.assertEqual(decision.kind, "catalog_only")
        self.assertEqual(decision.drug_search_terms, [" Drug A ", " "])
        self.assertEqual(
            model.full_messages_by_call[0],
            [
                {"role": "system", "content": QUESTION_TRIAGE_SYSTEM_PROMPT},
                {"role": "user", "content": "Drug A?"},
                {"role": "assistant", "content": "Earlier answer"},
                {"role": "user", "content": "and its strengths?"},
            ],
        )
        self.assertEqual(len(history), 2)
        self.assertEqual(
            model.calls[0][2],
            {
                "model": "test-model",
                "temperature": 0.01,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "QuestionTriage",
                        "schema": QuestionTriage.model_json_schema(),
                        "strict": True,
                    },
                },
            },
        )

    async def test_triage_has_an_explicit_combined_route_without_class_expansion(self) -> None:
        model = FakeChatModel(kind="clinical_and_drug", drug_search_terms=["SSRIs"])
        decision = await classify_question(
            question="guidance and stock",
            previous_messages=[],
            chat_completions_provider=model,
            model_name="test-model",
        )
        self.assertEqual(decision.kind, "clinical_and_drug")
        self.assertEqual(decision.drug_search_terms, ["SSRIs"])

    async def test_clinical_only_route_ignores_stray_stock_queries(self) -> None:
        model = FakeChatModel(kind="clinical", drug_search_terms=["do not search"])
        drugs = drug_agent({})
        _, answer = await collect_with_answer(
            ClinicalDrugOrchestrator(guideline_agent(), drugs, model)
        )
        self.assertEqual(answer.route, "clinical")
        drugs.stream_drug_answer_text.assert_not_called()


class FinalAnswerAssemblyTests(unittest.IsolatedAsyncioTestCase):
    """Answer assembly works directly, without running the graph or either agent."""

    def test_assembly_combines_stock_text_and_guideline_citations(self) -> None:
        for route, stock in (
            ("clinical", ""),
            ("clinical_and_drug", "stock table"),
            ("drug", "stock table"),
            ("drug", ""),
        ):
            with self.subTest(route=route, stock=stock):
                text, sources = build_final_answer(
                    route=route,
                    guideline_answer=None if route == "drug" else GUIDELINE_ANSWER,
                    catalog_section_text=stock,
                )
                expected = (
                    stock
                    if route == "drug"
                    else MARKED_GUIDELINE_TEXT + ("\n\n" + stock if stock else "")
                )
                self.assertEqual(text, expected)
                self.assertEqual(sources, [] if route == "drug" else GUIDELINE_ANSWER.citations)

    async def test_node_returns_the_assembled_answer_as_a_history_update(self) -> None:
        state = {
            "route": "clinical_and_drug",
            "guideline_answer": GUIDELINE_ANSWER,
            "catalog_section_text": "stock table",
        }
        original = state.copy()
        update = await finalize_answer(state)
        self.assertEqual(update["final_text"], MARKED_GUIDELINE_TEXT + "\n\nstock table")
        self.assertEqual(update["sources"], GUIDELINE_ANSWER.citations)
        self.assertEqual(len(update["conversation_messages"]), 1)
        self.assertEqual(update["conversation_messages"][0].content, update["final_text"])
        self.assertEqual(state, original)


class DrugStreamBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_result_answer_is_forwarded_on_both_stock_routes(self) -> None:
        for kind in ("catalog_only", "clinical_and_drug"):
            with self.subTest(kind=kind):
                model = FakeChatModel(kind=kind, drug_search_terms=["Drug A"])
                chunks, answer = await collect_with_answer(
                    ClinicalDrugOrchestrator(guideline_agent(), drug_agent({}), model)
                )
                prefix = GUIDELINE_CHUNKS if kind == "clinical_and_drug" else []
                separator = "\n\n" if kind == "clinical_and_drug" else ""
                stock = "No matching records were found."
                self.assertEqual(chunks, prefix + [separator + stock])
                self.assertEqual(
                    answer.text, (MARKED_GUIDELINE_TEXT + separator if prefix else "") + stock
                )
                self.assertEqual(len(model.calls), 1)

    async def test_history_saves_the_same_drug_answer_that_was_streamed(self) -> None:
        model = FakeChatModel(kind="clinical_and_drug", drug_search_terms=["Drug A"])
        orchestrator = ClinicalDrugOrchestrator(
            guideline_agent(), drug_agent({"Drug A": [drug("Lisinopril")]}), model
        )
        chunks, answer = await collect_with_answer(orchestrator, "first", "same-thread")
        self.assertEqual(
            answer.text, MARKED_GUIDELINE_TEXT + "".join(chunks[len(GUIDELINE_CHUNKS) :])
        )
        model.replies[QUESTION_TRIAGE_SYSTEM_PROMPT] = {"kind": "clinical", "drug_search_terms": []}
        await collect(orchestrator, "second", "same-thread")
        self.assertEqual(
            model.full_messages_by_call[1][1:-1],
            [{"role": "user", "content": "first"}, {"role": "assistant", "content": answer.text}],
        )


if __name__ == "__main__":
    unittest.main()
