"""Test the web interface's service layer: no browser, no network."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from healthcare_assistant_web.services.answer_streamer import AnswerStreamer
from healthcare_assistant_web.services.citation_text import citation_markers_as_superscript
from healthcare_assistant_web.services.conversation_log import ConversationLog
from healthcare_assistant_web.services.guideline_library import GuidelineLibrary
from healthcare_assistant_web.services.user_session import LazyUserSession
from src.clinical_guideline_agent.schema import FileCitation
from src.orchestrator import OrchestratorAnswer, OrchestratorUpdate


class FakeUpload:
    """Stands in for a NiceGUI upload: a name and an async save()."""

    def __init__(self, name: str, content: bytes = b"data") -> None:
        self.name = name
        self.content = content

    async def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.content)


def make_answer(text: str = "Answer", route: str = "clinical") -> OrchestratorAnswer:
    return OrchestratorAnswer(
        route=route,
        text=text,
        sources=[FileCitation(file_id="file_1", filename="ADHD NICE 2019.pdf")],
        guideline_answer=None,
        drug_classes=["ACE inhibitors"],
        drug_answer=None,
    )


class CitationTextTests(unittest.TestCase):
    def test_markers_become_superscripts(self) -> None:
        text = "Use an ACE inhibitor.${}^{[1]}$ Avoid NSAIDs.${}^{[2]}$"

        self.assertEqual(
            citation_markers_as_superscript(text),
            "Use an ACE inhibitor.<sup>[1]</sup> Avoid NSAIDs.<sup>[2]</sup>",
        )

    def test_text_without_markers_is_unchanged(self) -> None:
        self.assertEqual(citation_markers_as_superscript("No sources here."), "No sources here.")


class GuidelineLibraryTests(unittest.IsolatedAsyncioTestCase):
    def make_library(self):
        agent = MagicMock()
        agent.list_guideline_files = AsyncMock(return_value=["a file"])
        agent.add_guideline_files = AsyncMock()
        agent.remove_guideline_file = AsyncMock()
        return GuidelineLibrary(agent), agent

    async def test_pdf_and_markdown_files_are_indexed_under_their_own_names(self) -> None:
        library, agent = self.make_library()
        seen_file_names: list[str] = []

        async def capture_files(paths: list[Path]) -> None:
            # The files are deleted once the upload ends, so check them while they exist.
            seen_file_names.extend(path.name for path in paths if path.read_bytes() == b"data")

        agent.add_guideline_files.side_effect = capture_files

        result = await library.upload_files([FakeUpload("CKD KDIGO 2024.pdf"), FakeUpload("a.md")])

        self.assertEqual(result.indexed_names, ["CKD KDIGO 2024.pdf", "a.md"])
        self.assertEqual(result.skipped_names, [])
        self.assertEqual(seen_file_names, ["CKD KDIGO 2024.pdf", "a.md"])

    async def test_other_file_types_are_skipped(self) -> None:
        library, agent = self.make_library()

        result = await library.upload_files([FakeUpload("notes.docx"), FakeUpload("a.PDF")])

        self.assertEqual(result.indexed_names, ["a.PDF"])
        self.assertEqual(result.skipped_names, ["notes.docx"])

    async def test_nothing_is_sent_when_every_file_is_skipped(self) -> None:
        library, agent = self.make_library()

        result = await library.upload_files([FakeUpload("notes.docx")])

        self.assertEqual(result.indexed_names, [])
        agent.add_guideline_files.assert_not_awaited()

    async def test_two_files_with_the_same_name_do_not_overwrite_each_other(self) -> None:
        library, agent = self.make_library()
        contents: list[bytes] = []

        async def capture_contents(paths: list[Path]) -> None:
            contents.extend(path.read_bytes() for path in paths)

        agent.add_guideline_files.side_effect = capture_contents

        await library.upload_files([FakeUpload("a.pdf", b"first"), FakeUpload("a.pdf", b"second")])

        self.assertEqual(contents, [b"first", b"second"])

    async def test_list_and_delete_go_to_the_agent(self) -> None:
        library, agent = self.make_library()

        self.assertEqual(await library.list_files(), ["a file"])
        await library.delete_file("file_1")

        agent.remove_guideline_file.assert_awaited_once_with("file_1")


class ConversationLogTests(unittest.TestCase):
    def test_turns_are_appended_to_one_file_per_session(self) -> None:
        with TemporaryDirectory() as directory:
            log = ConversationLog(Path(directory))

            log.save_turn("session_1", "req_1", "First?", make_answer("One"))
            log.save_turn("session_1", "req_2", "Second?", make_answer("Two"))

            saved = json.loads((Path(directory) / "session_1.json").read_text(encoding="utf-8"))
            self.assertEqual([turn["question"] for turn in saved["turns"]], ["First?", "Second?"])
            self.assertEqual(saved["turns"][0]["sources"], ["ADHD NICE 2019.pdf"])
            self.assertEqual(saved["turns"][0]["drug_classes"], ["ACE inhibitors"])


class AnswerStreamerTests(unittest.IsolatedAsyncioTestCase):
    async def test_updates_pass_through_and_only_the_final_answer_is_saved(self) -> None:
        answer = make_answer()

        thread_ids_received: list[str] = []

        async def orchestrator_updates(question: str, conversation_thread_id: str):
            thread_ids_received.append(conversation_thread_id)
            yield OrchestratorUpdate(type="text", text="Hel")
            yield OrchestratorUpdate(type="text", text="lo")
            yield OrchestratorUpdate(type="completed", answer=answer)

        orchestrator = MagicMock()
        orchestrator.stream_clinical_drug_answer_updates = orchestrator_updates
        conversation_log = MagicMock()
        streamer = AnswerStreamer(orchestrator, "session_1", conversation_log)

        updates = [update async for update in streamer.stream_answer("Hi?")]

        self.assertEqual([update.type for update in updates], ["text", "text", "completed"])
        # The tab's session id is the conversation thread id, so follow-ups see history.
        self.assertEqual(thread_ids_received, ["session_1"])
        conversation_log.save_turn.assert_called_once()
        saved_arguments = conversation_log.save_turn.call_args.args
        self.assertEqual(saved_arguments[0], "session_1")
        self.assertEqual(saved_arguments[2:], ("Hi?", answer))


class LazyUserSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_nothing_connects_until_the_session_is_needed(self) -> None:
        manager = MagicMock()
        manager.get = AsyncMock()

        LazyUserSession(manager)

        manager.get.assert_not_awaited()

    async def test_the_session_is_created_once_and_reused(self) -> None:
        shared_services = SimpleNamespace(
            guidelines_store_id="vs_1",
            drug_agent=MagicMock(),
            chat_completions_provider=MagicMock(),
        )
        manager = MagicMock()
        manager.get = AsyncMock(return_value=shared_services)
        lazy_session = LazyUserSession(manager)

        with unittest.mock.patch(
            "healthcare_assistant_web.services.user_session.ClinicalGuidelineAgent"
        ) as agent_class:
            agent_class.return_value.close = AsyncMock()
            first = await lazy_session.get()
            second = await lazy_session.get()
            await lazy_session.close()

        self.assertIs(first, second)
        agent_class.assert_called_once_with(guidelines_store_id="vs_1")
        agent_class.return_value.close.assert_awaited_once()

    async def test_a_failed_connection_is_retried_on_the_next_request(self) -> None:
        manager = MagicMock()
        manager.get = AsyncMock(side_effect=[RuntimeError("down"), RuntimeError("still down")])
        lazy_session = LazyUserSession(manager)

        for _ in range(2):
            with self.assertRaises(RuntimeError):
                await lazy_session.get()

        self.assertEqual(manager.get.await_count, 2)


if __name__ == "__main__":
    unittest.main()
