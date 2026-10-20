"""Exercise the new guideline package with SDK-shaped objects, without API calls."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.adapters.vector_store.openai_vector_store_adapter import OpenAIVectorStoreAdapter
from src.clinical_guideline_agent import ClinicalGuidelineAgent
from src.clinical_guideline_agent.vector_store.guidelines_store import GuidelinesStore


def page(*items):
    async def iterate():
        for item in items:
            yield item

    return iterate()


def response(response_id="resp_1"):
    annotation = SimpleNamespace(
        type="file_citation", file_id="file_1", filename="guideline.pdf", index=12
    )
    content = SimpleNamespace(
        type="output_text", text="Follow the guideline.", annotations=[annotation]
    )
    message = SimpleNamespace(type="message", content=[content])
    return SimpleNamespace(
        id=response_id, status="completed", output_text="Follow the guideline.", output=[message]
    )


class ClinicalGuidelineAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_uses_file_search_and_chains_completed_responses(self) -> None:
        provider = MagicMock()
        provider.execute_responses_api_raw = AsyncMock(side_effect=[response(), response("resp_2")])
        agent = ClinicalGuidelineAgent(
            responses_api_provider=provider,
            guidelines_vector_store_adapter=MagicMock(),
            model_name="test-model",
            guidelines_store_id="vs_1",
        )

        first = await agent.answer_guideline_question("What do the guidelines say?")
        second = await agent.answer_guideline_question("Can you explain that?")

        self.assertEqual(first.citations[0].filename, "guideline.pdf")
        self.assertEqual(second.response_id, "resp_2")
        first_request = provider.execute_responses_api_raw.await_args_list[0].kwargs
        second_request = provider.execute_responses_api_raw.await_args_list[1].kwargs
        self.assertEqual(
            first_request["tools"], [{"type": "file_search", "vector_store_ids": ["vs_1"]}]
        )
        self.assertTrue(first_request["store"])
        self.assertEqual(second_request["previous_response_id"], "resp_1")
        self.assertEqual(second_request["instructions"], first_request["instructions"])

    async def test_stream_uses_event_types_and_finishes_with_citations(self) -> None:
        provider = MagicMock()
        provider.execute_responses_api_raw = AsyncMock(
            return_value=page(
                SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_1")),
                SimpleNamespace(type="response.output_text.delta", delta="Follow "),
                SimpleNamespace(type="response.completed", response=response()),
            )
        )
        agent = ClinicalGuidelineAgent(
            responses_api_provider=provider,
            guidelines_vector_store_adapter=MagicMock(),
            model_name="test-model",
            guidelines_store_id="vs_1",
        )

        updates = [update async for update in agent.stream_guideline_answer_updates("Question")]

        self.assertEqual([update.type for update in updates], ["text", "completed"])
        self.assertEqual(updates[0].text, "Follow ")
        self.assertEqual(updates[-1].answer.citations[0].file_id, "file_1")
        self.assertEqual(
            agent.responses_api_conversation_state.latest_responses_api_response_id,
            "resp_1",
        )

    async def test_stream_error_does_not_advance_chain(self) -> None:
        provider = MagicMock()
        provider.execute_responses_api_raw = AsyncMock(
            return_value=page(SimpleNamespace(type="error", message="service unavailable"))
        )
        agent = ClinicalGuidelineAgent(
            responses_api_provider=provider,
            guidelines_vector_store_adapter=MagicMock(),
            model_name="test-model",
            guidelines_store_id="vs_1",
        )

        with self.assertRaisesRegex(RuntimeError, "service unavailable"):
            async for _ in agent.stream_guideline_answer_updates("Question"):
                pass
        self.assertIsNone(agent.responses_api_conversation_state.latest_responses_api_response_id)

    async def test_ready_vector_store_is_reused(self) -> None:
        adapter = MagicMock()
        adapter.list_stores = AsyncMock(
            return_value=[SimpleNamespace(name="clinical_guidelines", id="vs_1")]
        )
        adapter.retrieve_store = AsyncMock(return_value=SimpleNamespace(status="completed"))
        adapter.list_files = AsyncMock(
            return_value=[SimpleNamespace(id="file_1", status="completed")]
        )
        store = GuidelinesStore(adapter)

        self.assertEqual(await store.ensure_ready("unused-directory"), "vs_1")
        adapter.create_store.assert_not_called()

    async def test_new_store_waits_for_file_indexing(self) -> None:
        adapter = MagicMock()
        adapter.list_stores = AsyncMock(return_value=[])
        adapter.create_store = AsyncMock(return_value=SimpleNamespace(id="vs_new"))
        adapter.retrieve_store = AsyncMock(return_value=SimpleNamespace(status="completed"))
        adapter.upload_and_poll = AsyncMock(return_value=SimpleNamespace(status="completed"))
        adapter.list_files = AsyncMock(
            return_value=[SimpleNamespace(id="file_1", status="completed")]
        )
        store = GuidelinesStore(adapter)

        with TemporaryDirectory() as temporary_directory:
            Path(temporary_directory, "guideline.md").write_text("Guideline", encoding="utf-8")
            store_id = await store.ensure_ready(temporary_directory)

        self.assertEqual(store_id, "vs_new")
        adapter.upload_and_poll.assert_awaited_once()
        adapter.retrieve_store.assert_awaited_once_with("vs_new")

    async def test_vector_store_adapter_uses_sdk(self) -> None:
        client = MagicMock()
        client.vector_stores.list = AsyncMock(return_value=page(SimpleNamespace(id="vs_1")))
        adapter = OpenAIVectorStoreAdapter(client=client)

        stores = await adapter.list_stores()

        self.assertEqual(stores[0].id, "vs_1")
        client.vector_stores.list.assert_awaited_once()
