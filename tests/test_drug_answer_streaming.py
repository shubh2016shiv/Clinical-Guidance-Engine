"""The drug agent sends a table to one LLM call and forwards its live prose."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.drugs_agent import DrugAgent
from src.orchestrator import ClinicalDrugOrchestrator
from src.orchestrator.nodes import search_catalog
from src.prompts.drug_system_prompt import DRUG_STREAM_SYSTEM_PROMPT


def create_agent(rows_by_query, stream):
    async def search(**kwargs):
        return [SimpleNamespace(fields=row) for row in rows_by_query.get(kwargs["drug_name"], [])]

    searcher = SimpleNamespace(search_drug_records=AsyncMock(side_effect=search))
    classifier = SimpleNamespace(identify_drug_class=AsyncMock(return_value="SSRIs"))
    provider = SimpleNamespace(execute_chat_completion=AsyncMock(return_value=stream))
    agent = DrugAgent(
        searcher, provider, chat_completion_model_name="test-model", drug_classifier=classifier
    )
    return agent, provider


class DrugAnswerStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_agent_and_graph_stream_and_remember_the_same_answer(self):
        async def stream():
            yield "Sertraline is "
            yield "listed in the catalog."

        agent, provider = create_agent({"SSRIs": [{"drug_name": "Sertraline"}]}, stream())

        async def complete(messages, **options):
            if options.get("stream"):
                return stream()
            return SimpleNamespace(
                finish_reason="stop",
                content=json.dumps({"kind": "catalog_only", "drug_search_terms": ["SSRIs"]}),
            )

        provider.execute_chat_completion.side_effect = complete
        orchestrator = ClinicalDrugOrchestrator(SimpleNamespace(), agent, provider)
        updates = [
            update
            async for update in orchestrator.stream_clinical_drug_answer_updates(
                "Which SSRIs are available?", "streaming-thread"
            )
        ]
        self.assertEqual(
            [update.text for update in updates if update.type == "text"],
            ["Sertraline is ", "listed in the catalog."],
        )
        self.assertEqual([update.type for update in updates].count("completed"), 1)
        answer = updates[-1].answer
        self.assertEqual(answer.text, "Sertraline is listed in the catalog.")
        self.assertEqual(provider.execute_chat_completion.await_count, 2)  # triage + one answer
        saved = await orchestrator.graph.aget_state(
            {"configurable": {"thread_id": "streaming-thread"}}
        )
        self.assertEqual(saved.values["conversation_messages"][-1].content, answer.text)

    async def test_one_answer_call_receives_a_table_and_streams_before_completion(self):
        next_chunk_requested = asyncio.Event()

        async def stream():
            yield "Listed in the catalog: "
            next_chunk_requested.set()
            yield "Sertraline."

        agent, provider = create_agent(
            {"SSRIs": [{"drug_name": "Sertraline", "formulation": "", "dosage_strengths": "[]"}]},
            stream(),
        )
        answer = agent.stream_drug_answer_text(
            question="Which SSRIs are listed?", drug_search_terms=["SSRIs"]
        )
        self.assertEqual(await anext(answer), "Listed in the catalog: ")
        self.assertFalse(next_chunk_requested.is_set())
        self.assertEqual([chunk async for chunk in answer], ["Sertraline."])
        provider.execute_chat_completion.assert_awaited_once()
        messages = provider.execute_chat_completion.call_args.args[0]
        options = provider.execute_chat_completion.call_args.kwargs
        self.assertEqual(messages[0]["content"], DRUG_STREAM_SYSTEM_PROMPT)
        self.assertIn('Question: "Which SSRIs are listed?"', messages[1]["content"])
        self.assertIn("| drug_name", messages[1]["content"])
        self.assertIn('"Sertraline"', messages[1]["content"])
        self.assertNotIn("formulation", messages[1]["content"])
        self.assertTrue(options["stream"])
        self.assertNotIn("response_format", options)
        agent.drug_classifier.identify_drug_class.assert_awaited_once_with("SSRIs")
        agent.drug_searcher.search_drug_records.assert_awaited_once_with(
            drug_name="SSRIs", drug_class="SSRIs", max_drug_results=None
        )

    async def test_query_batch_is_ordered_and_identical_rows_are_deduplicated(self):
        async def stream():
            yield "An answer."

        shared = {"drug_name": "Sertraline", "formulation": "tablet"}
        distinct = {"drug_name": "Sertraline", "formulation": "solution"}
        agent, provider = create_agent({"a": [shared], "b": [shared, distinct]}, stream())
        later_search_finished = asyncio.Event()
        original_search = agent.drug_searcher.search_drug_records.side_effect

        async def search(**kwargs):
            if kwargs["drug_name"] == "a":
                await later_search_finished.wait()
            else:
                later_search_finished.set()
            return await original_search(**kwargs)

        agent.drug_searcher.search_drug_records.side_effect = search
        await asyncio.wait_for(self.collect(agent, ["a", "b"]), timeout=5)
        content = provider.execute_chat_completion.call_args.args[0][1]["content"]
        self.assertEqual(content.count('"tablet"'), 1)
        self.assertEqual(content.count('"solution"'), 1)
        self.assertLess(content.index('"tablet"'), content.index('"solution"'))

    async def collect(self, agent, terms):
        return [
            chunk
            async for chunk in agent.stream_drug_answer_text(
                question="Question", drug_search_terms=terms
            )
        ]

    async def test_empty_retrieval_still_uses_the_llm_to_answer(self):
        async def stream():
            yield "No relevant records were found."

        agent, provider = create_agent({}, stream())
        self.assertEqual(
            await self.collect(agent, ["missing"]), ["No relevant records were found."]
        )
        self.assertIn(
            "No drug records were retrieved.",
            provider.execute_chat_completion.call_args.args[0][1]["content"],
        )

    async def test_search_failure_prevents_answer_generation(self):
        agent, provider = create_agent({}, None)
        agent.drug_searcher.search_drug_records.side_effect = RuntimeError("search failed")
        with self.assertRaisesRegex(RuntimeError, "search failed"):
            await self.collect(agent, ["a"])
        provider.execute_chat_completion.assert_not_awaited()

    async def test_empty_stream_is_an_error_and_empty_chunks_are_ignored(self):
        async def stream():
            yield ""

        agent, _ = create_agent({}, stream())
        with self.assertRaisesRegex(RuntimeError, "without any answer text"):
            await self.collect(agent, ["a"])

    async def test_partial_stream_failure_propagates(self):
        failure = RuntimeError("connection lost")

        async def stream():
            yield "partial answer"
            raise failure

        agent, _ = create_agent({}, stream())
        answer = agent.stream_drug_answer_text(question="Question", drug_search_terms=["a"])
        self.assertEqual(await anext(answer), "partial answer")
        with self.assertRaises(RuntimeError) as raised:
            await anext(answer)
        self.assertIs(raised.exception, failure)


class DrugNodeStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_chunks_are_forwarded_immediately_with_one_separator(self):
        for route, separator in (("drug", ""), ("clinical_and_drug", "\n\n")):
            with self.subTest(route=route):
                emitted = []

                async def stream(observed=emitted, prefix=separator, **kwargs):
                    self.assertEqual(
                        kwargs, {"question": "question", "drug_search_terms": ["SSRIs"]}
                    )
                    yield ""
                    self.assertEqual(observed, [])
                    yield "First "
                    self.assertEqual(observed, [prefix + "First "])
                    yield "second."

                agent = SimpleNamespace(stream_drug_answer_text=stream)
                with (
                    patch(
                        "src.orchestrator.nodes.node_03_drug_catalog_answer.send_answer_text",
                        side_effect=emitted.append,
                    ),
                    patch("src.orchestrator.nodes.node_03_drug_catalog_answer.send_status"),
                ):
                    update = await search_catalog(
                        {
                            "clinical_question": "question",
                            "drug_search_terms": ["SSRIs"],
                            "route": route,
                        },
                        SimpleNamespace(context=SimpleNamespace(drug_agent=agent)),
                    )
                self.assertEqual(emitted, [separator + "First ", "second."])
                self.assertEqual(update, {"catalog_section_text": "First second."})

    async def test_failed_or_empty_stream_does_not_complete_the_graph(self):
        for partial in (False, True):
            with self.subTest(partial=partial):

                async def stream(should_fail=partial, **kwargs):
                    if should_fail:
                        yield "Partial"
                        raise RuntimeError("stream failed")
                    yield ""

                with (
                    patch(
                        "src.orchestrator.nodes.node_03_drug_catalog_answer.send_answer_text"
                    ) as send,
                    patch("src.orchestrator.nodes.node_03_drug_catalog_answer.send_status"),
                    self.assertRaisesRegex(RuntimeError, "stream failed|without any answer text"),
                ):
                    await search_catalog(
                        {
                            "clinical_question": "question",
                            "drug_search_terms": ["a"],
                            "route": "drug",
                        },
                        SimpleNamespace(
                            context=SimpleNamespace(
                                drug_agent=SimpleNamespace(stream_drug_answer_text=stream)
                            )
                        ),
                    )
                self.assertEqual(send.call_count, int(partial))
