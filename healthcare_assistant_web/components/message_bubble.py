"""One message in the conversation: the user's question or the assistant's answer."""

from nicegui import ui

from src.clinical_guideline_agent.schema import FileCitation
from src.orchestrator import OrchestratorAnswer

from ..services.citation_text import citation_markers_as_superscript
from .citations_block import build_sources_block

# What to call an answer, based on which agents the orchestrator actually used.
ANSWER_KIND_BY_ROUTE = {
    "clinical": "Guideline answer",
    "clinical_and_drug": "Guidelines and drug catalog",
    "drug": "Drug catalog",
}


def add_user_message(messages_column: ui.column, question: str) -> None:
    with messages_column, ui.row().classes("w-full justify-end items-start gap-2 no-wrap"):
        ui.label(question).classes("user-message")
        ui.icon("person", size="1rem").classes("mt-2").style("color: var(--text-faint)")


class AssistantMessage:
    """An answer that fills in while it is being written."""

    def __init__(self, messages_column: ui.column) -> None:
        with messages_column, ui.column().classes("assistant-card gap-3"):
            with ui.row().classes("items-center gap-2 no-wrap"):
                ui.icon("auto_awesome", size="1.1rem").style("color: var(--brand)")
                self.kind_label = ui.label("Thinking...").classes("answer-kind")
                self.badge_area = ui.row().classes("items-center gap-2")

            # Dots show until the first piece of text arrives.
            self.waiting_dots = ui.spinner("dots", size="1.6rem", color="primary")
            self.answer_text = ui.markdown().classes("answer-body w-full")
            self.sources_area = ui.column().classes("w-full gap-1")

    def show_text(self, text: str) -> None:
        """Show the answer written so far."""
        self.waiting_dots.set_visibility(False)
        self.answer_text.set_content(citation_markers_as_superscript(text))

    def finish(self, answer: OrchestratorAnswer) -> None:
        """Replace the streamed text with the final answer, and list its sources."""
        self.show_text(answer.text)
        self.kind_label.set_text(ANSWER_KIND_BY_ROUTE.get(answer.route, "Answer"))
        self._add_badge(answer.sources)
        with self.sources_area:
            build_sources_block(answer.sources)

    def show_error(self, message: str) -> None:
        self.waiting_dots.set_visibility(False)
        self.kind_label.set_text("Could not answer")
        self.answer_text.set_content(message)

    def _add_badge(self, sources: list[FileCitation]) -> None:
        if not sources:
            return
        file_count = len({source.file_id for source in sources})
        guideline_word = "guideline" if file_count == 1 else "guidelines"
        with self.badge_area:
            ui.label(f"Based on {file_count} {guideline_word}").classes("answer-badge")
