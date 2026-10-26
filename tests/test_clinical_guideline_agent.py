"""Exercise the new guideline package with SDK-shaped objects, without API calls."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from openai.types.responses.response import IncompleteDetails, Response
from openai.types.responses.response_error import ResponseError
from openai.types.responses.response_failed_event import ResponseFailedEvent
from openai.types.responses.response_incomplete_event import ResponseIncompleteEvent

from src.adapters.vector_store.openai_vector_store import OpenAIVectorStoreClient
from src.clinical_guideline_agent import ClinicalGuidelineAgent
from src.clinical_guideline_agent.tools.file_search import file_search_tool
from src.clinical_guideline_agent.vector_store.guidelines_store import GuidelinesStore


def page(*items):
    async def iterate():
        for item in items:
            yield item

    return iterate()


def minimal_response(**overrides):
    """A real openai-python Response object, not a stand-in, so field access matches the SDK."""
    defaults = dict(
        id="resp_1",
        created_at=0.0,
        model="gpt-4o",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )
    return Response(**{**defaults, **overrides})


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
        provider.execute_responses_api = AsyncMock(side_effect=[response(), response("resp_2")])
        agent = ClinicalGuidelineAgent(
            responses_api_provider=provider,
            openai_vector_store_client=MagicMock(),
            model_name="test-model",
            guidelines_store_id="vs_1",
        )

        first = await agent.answer_guideline_question("What do the guidelines say?")
        second = await agent.answer_guideline_question("Can you explain that?")

        self.assertEqual(first.citations[0].filename, "guideline.pdf")
        self.assertEqual(second.response_id, "resp_2")
        first_request = provider.execute_responses_api.await_args_list[0].kwargs
        second_request = provider.execute_responses_api.await_args_list[1].kwargs
        self.assertEqual(
            first_request["tools"], [{"type": "file_search", "vector_store_ids": ["vs_1"]}]
        )
        self.assertTrue(first_request["store"])
        self.assertEqual(second_request["previous_response_id"], "resp_1")
        self.assertEqual(second_request["instructions"], first_request["instructions"])

    async def test_stream_uses_event_types_and_finishes_with_citations(self) -> None:
        provider = MagicMock()
        provider.execute_responses_api = AsyncMock(
            return_value=page(
                SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_1")),
                SimpleNamespace(type="response.output_text.delta", delta="Follow "),
                SimpleNamespace(type="response.completed", response=response()),
            )
        )
        agent = ClinicalGuidelineAgent(
            responses_api_provider=provider,
            openai_vector_store_client=MagicMock(),
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
        provider.execute_responses_api = AsyncMock(
            return_value=page(SimpleNamespace(type="error", message="service unavailable"))
        )
        agent = ClinicalGuidelineAgent(
            responses_api_provider=provider,
            openai_vector_store_client=MagicMock(),
            model_name="test-model",
            guidelines_store_id="vs_1",
        )

        with self.assertRaisesRegex(RuntimeError, "service unavailable"):
            async for _ in agent.stream_guideline_answer_updates("Question"):
                pass
        self.assertIsNone(agent.responses_api_conversation_state.latest_responses_api_response_id)

    async def test_stream_failed_event_surfaces_the_real_response_error(self) -> None:
        """response.failed carries its reason under response.error, not on the event."""
        failed_response = minimal_response(
            status="failed",
            error=ResponseError(code="server_error", message="internal server error"),
        )
        failed_event = ResponseFailedEvent(
            response=failed_response, sequence_number=1, type="response.failed"
        )
        provider = MagicMock()
        provider.execute_responses_api = AsyncMock(return_value=page(failed_event))
        agent = ClinicalGuidelineAgent(
            responses_api_provider=provider,
            openai_vector_store_client=MagicMock(),
            model_name="test-model",
            guidelines_store_id="vs_1",
        )

        with self.assertRaisesRegex(RuntimeError, "internal server error"):
            async for _ in agent.stream_guideline_answer_updates("Question"):
                pass

    async def test_stream_incomplete_event_surfaces_the_real_reason(self) -> None:
        """response.incomplete carries its reason under response.incomplete_details."""
        incomplete_response = minimal_response(
            status="incomplete",
            incomplete_details=IncompleteDetails(reason="max_output_tokens"),
        )
        incomplete_event = ResponseIncompleteEvent(
            response=incomplete_response, sequence_number=1, type="response.incomplete"
        )
        provider = MagicMock()
        provider.execute_responses_api = AsyncMock(return_value=page(incomplete_event))
        agent = ClinicalGuidelineAgent(
            responses_api_provider=provider,
            openai_vector_store_client=MagicMock(),
            model_name="test-model",
            guidelines_store_id="vs_1",
        )

        with self.assertRaisesRegex(RuntimeError, "max_output_tokens"):
            async for _ in agent.stream_guideline_answer_updates("Question"):
                pass

    async def test_answer_rejects_a_failed_response(self) -> None:
        """The non-streaming path reads the same nested error fields."""
        provider = MagicMock()
        provider.execute_responses_api = AsyncMock(
            return_value=minimal_response(
                status="failed",
                error=ResponseError(code="server_error", message="internal server error"),
            )
        )
        agent = ClinicalGuidelineAgent(
            responses_api_provider=provider,
            openai_vector_store_client=MagicMock(),
            model_name="test-model",
            guidelines_store_id="vs_1",
        )

        with self.assertRaisesRegex(RuntimeError, "internal server error"):
            await agent.answer_guideline_question("Question")

    async def test_answer_rejects_an_incomplete_response(self) -> None:
        provider = MagicMock()
        provider.execute_responses_api = AsyncMock(
            return_value=minimal_response(
                status="incomplete",
                incomplete_details=IncompleteDetails(reason="content_filter"),
            )
        )
        agent = ClinicalGuidelineAgent(
            responses_api_provider=provider,
            openai_vector_store_client=MagicMock(),
            model_name="test-model",
            guidelines_store_id="vs_1",
        )

        with self.assertRaisesRegex(RuntimeError, "content_filter"):
            await agent.answer_guideline_question("Question")

    async def test_ready_vector_store_is_reused(self) -> None:
        openai_vector_store_client = MagicMock()
        openai_vector_store_client.list_vector_stores = AsyncMock(
            return_value=[SimpleNamespace(name="clinical_guidelines", id="vs_1")]
        )
        openai_vector_store_client.retrieve_vector_store = AsyncMock(
            return_value=SimpleNamespace(status="completed")
        )
        openai_vector_store_client.list_vector_store_files = AsyncMock(
            return_value=[SimpleNamespace(id="file_1", status="completed")]
        )
        store = GuidelinesStore(openai_vector_store_client)

        self.assertEqual(await store.ensure_ready("unused-directory"), "vs_1")
        openai_vector_store_client.create_vector_store.assert_not_called()

    async def test_new_store_waits_for_file_indexing(self) -> None:
        openai_vector_store_client = MagicMock()
        openai_vector_store_client.list_vector_stores = AsyncMock(return_value=[])
        openai_vector_store_client.create_vector_store = AsyncMock(
            return_value=SimpleNamespace(id="vs_new")
        )
        openai_vector_store_client.retrieve_vector_store = AsyncMock(
            return_value=SimpleNamespace(status="completed")
        )
        openai_vector_store_client.upload_files_and_wait = AsyncMock(
            return_value=SimpleNamespace(file_counts=SimpleNamespace(failed=0, total=1))
        )
        openai_vector_store_client.list_vector_store_files = AsyncMock(
            return_value=[SimpleNamespace(id="file_1", status="completed")]
        )
        store = GuidelinesStore(openai_vector_store_client)

        with TemporaryDirectory() as temporary_directory:
            Path(temporary_directory, "guideline.md").write_text("Guideline", encoding="utf-8")
            store_id = await store.ensure_ready(temporary_directory)

        self.assertEqual(store_id, "vs_new")
        openai_vector_store_client.upload_files_and_wait.assert_awaited_once()
        openai_vector_store_client.retrieve_vector_store.assert_awaited_once_with("vs_new")

    async def test_missing_guidelines_directory_does_not_create_store(self) -> None:
        openai_vector_store_client = MagicMock()
        openai_vector_store_client.list_vector_stores = AsyncMock(return_value=[])
        store = GuidelinesStore(openai_vector_store_client)

        with TemporaryDirectory() as temporary_directory:
            missing_directory = Path(temporary_directory, "missing")
            with self.assertRaisesRegex(ValueError, "Guideline directory does not exist"):
                await store.ensure_ready(missing_directory)

        openai_vector_store_client.create_vector_store.assert_not_called()

    async def test_failed_guideline_file_is_not_treated_as_ready(self) -> None:
        openai_vector_store_client = MagicMock()
        openai_vector_store_client.retrieve_vector_store = AsyncMock(
            return_value=SimpleNamespace(status="completed")
        )
        openai_vector_store_client.list_vector_store_files = AsyncMock(
            return_value=[SimpleNamespace(id="file_1", status="failed")]
        )

        with self.assertRaisesRegex(ValueError, "Guideline file file_1 is failed"):
            await GuidelinesStore(openai_vector_store_client).check_ready("vs_1")

    async def test_openai_vector_store_client_uses_sdk(self) -> None:
        openai_client = MagicMock()
        openai_client.vector_stores.list = AsyncMock(return_value=page(SimpleNamespace(id="vs_1")))
        vector_store_client = OpenAIVectorStoreClient(openai_client=openai_client)

        vector_stores = await vector_store_client.list_vector_stores()

        self.assertEqual(vector_stores[0].id, "vs_1")
        openai_client.vector_stores.list.assert_awaited_once()


class GuidelinesStoreFreshStartTests(unittest.IsolatedAsyncioTestCase):
    def make_client(self, listed=()):
        client = MagicMock()
        client.list_vector_stores = AsyncMock(return_value=list(listed))
        client.create_vector_store = AsyncMock(return_value=SimpleNamespace(id="vs_new"))
        client.delete_vector_store = AsyncMock(return_value=True)
        return client

    async def test_start_fresh_returns_a_new_empty_store(self) -> None:
        client = self.make_client()
        store_id = await GuidelinesStore(client).start_fresh()

        self.assertEqual(store_id, "vs_new")
        client.create_vector_store.assert_awaited_once_with("clinical_guidelines")

    async def test_start_fresh_deletes_stores_left_by_earlier_runs(self) -> None:
        client = self.make_client(
            listed=[
                SimpleNamespace(id="vs_old", name="clinical_guidelines"),
                SimpleNamespace(id="vs_other", name="something_else"),
            ]
        )

        await GuidelinesStore(client).start_fresh()

        client.delete_vector_store.assert_awaited_once_with("vs_old")

    async def test_start_fresh_deletes_a_known_previous_id_missing_from_the_list(self) -> None:
        """The list endpoint is eventually consistent, so a just-created store can be absent."""
        client = self.make_client(listed=[])

        await GuidelinesStore(client).start_fresh("vs_just_created")

        client.delete_vector_store.assert_awaited_once_with("vs_just_created")

    async def test_start_fresh_does_not_delete_the_previous_id_twice(self) -> None:
        client = self.make_client(
            listed=[SimpleNamespace(id="vs_previous", name="clinical_guidelines")]
        )

        await GuidelinesStore(client).start_fresh("vs_previous")

        client.delete_vector_store.assert_awaited_once_with("vs_previous")

    async def test_add_files_rejects_an_empty_upload(self) -> None:
        with self.assertRaisesRegex(ValueError, "No guideline files"):
            await GuidelinesStore(self.make_client()).add_files("vs_1", [])

    async def test_add_files_reports_failed_uploads(self) -> None:
        client = self.make_client()
        client.upload_files_and_wait = AsyncMock(
            return_value=SimpleNamespace(file_counts=SimpleNamespace(failed=1, total=2))
        )

        with self.assertRaisesRegex(RuntimeError, "1 of 2"):
            await GuidelinesStore(client).add_files("vs_1", [Path("a.pdf")])

    async def test_agent_start_fresh_passes_its_previous_store_id(self) -> None:
        agent = ClinicalGuidelineAgent(
            responses_api_provider=MagicMock(),
            openai_vector_store_client=MagicMock(),
            model_name="test-model",
            guidelines_store_id="vs_previous",
        )
        agent.guidelines_store.start_fresh = AsyncMock(return_value="vs_new")

        self.assertEqual(await agent.start_fresh_guidelines_store(), "vs_new")
        agent.guidelines_store.start_fresh.assert_awaited_once_with("vs_previous")
        self.assertEqual(agent.guidelines_store_id, "vs_new")

    async def test_agent_passes_display_names_down_to_the_upload(self) -> None:
        client = self.make_client()
        client.upload_files_and_wait = AsyncMock(
            return_value=SimpleNamespace(file_counts=SimpleNamespace(failed=0, total=1))
        )
        agent = ClinicalGuidelineAgent(
            responses_api_provider=MagicMock(),
            openai_vector_store_client=client,
            model_name="test-model",
            guidelines_store_id="vs_1",
        )
        stored = Path("31a98535.pdf")

        await agent.add_guideline_files([stored], display_names={stored: "hypertension.pdf"})

        client.upload_files_and_wait.assert_awaited_once_with(
            "vs_1", [stored], display_names={stored: "hypertension.pdf"}
        )

    async def test_agent_add_guideline_files_requires_a_store(self) -> None:
        agent = ClinicalGuidelineAgent(
            responses_api_provider=MagicMock(),
            openai_vector_store_client=MagicMock(),
            model_name="test-model",
        )

        with self.assertRaisesRegex(RuntimeError, "prepare_guidelines_store"):
            await agent.add_guideline_files([Path("a.pdf")])


class FileSearchToolTests(unittest.TestCase):
    def test_max_results_within_range_is_set(self) -> None:
        self.assertEqual(file_search_tool("vs_1", max_results=50)["max_num_results"], 50)
        self.assertEqual(file_search_tool("vs_1", max_results=1)["max_num_results"], 1)

    def test_max_results_omitted_when_not_given(self) -> None:
        self.assertNotIn("max_num_results", file_search_tool("vs_1"))

    def test_max_results_above_fifty_is_rejected(self) -> None:
        """The API itself rejects max_num_results > 50; fail locally instead of round-tripping."""
        with self.assertRaisesRegex(ValueError, "between 1 and 50"):
            file_search_tool("vs_1", max_results=51)

    def test_max_results_below_one_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 50"):
            file_search_tool("vs_1", max_results=0)

    def test_missing_vector_store_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "vector store ID"):
            file_search_tool("")
