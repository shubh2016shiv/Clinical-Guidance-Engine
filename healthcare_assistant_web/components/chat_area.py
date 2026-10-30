"""The conversation: the start screen, the messages, and the box where questions are typed."""

from nicegui import ui

from src.logs import get_component_logger

from ..services.user_session import LazyUserSession
from .message_bubble import AssistantMessage, add_user_message
from .start_screen import build_start_screen

logger = get_component_logger("HealthcareAssistantWeb")

ANSWER_FAILED_MESSAGE = "Sorry, something went wrong while answering. Please try again."


class ChatArea:
    """Shows the conversation and sends each question through the orchestrator."""

    def __init__(self, lazy_user_session: LazyUserSession) -> None:
        self.lazy_user_session = lazy_user_session
        self._answer_in_progress = False

    def build_messages(self) -> None:
        """Draw the start screen and the (initially empty) list of messages."""
        self.start_screen = build_start_screen(self.ask)
        self.messages_column = ui.column().classes("chat-column gap-5 px-4 pb-4")

    def build_input_bar(self) -> None:
        """Draw the question box. Call this inside the page footer."""
        with ui.column().classes("chat-column items-center gap-1"):
            with ui.row().classes("input-bar w-full items-center no-wrap gap-1"):
                self.question_input = (
                    ui.input(placeholder="Ask about a guideline, drug, or clinical question...")
                    .props("borderless dense")
                    .classes("grow text-sm")
                    .on("keydown.enter", self._send_typed_question)
                )
                self.send_button = (
                    ui.button(icon="arrow_upward", on_click=self._send_typed_question)
                    .props("round dense unelevated")
                    .classes("send-button")
                )
            ui.label(
                "AI can make mistakes. Please verify important information with clinical sources."
            ).classes("disclaimer")

    async def start_new_chat(self) -> None:
        """Clear the conversation and go back to the start screen."""
        if self._answer_in_progress:
            ui.notify("Please wait for the current answer to finish.", type="warning")
            return
        await self.lazy_user_session.restart()
        self.messages_column.clear()
        self.start_screen.set_visibility(True)
        self.question_input.value = ""

    async def _send_typed_question(self) -> None:
        question = self.question_input.value.strip()
        if not question:
            return
        self.question_input.value = ""
        await self.ask(question)

    async def ask(self, question: str) -> None:
        """Show the question, then stream the answer into a new message."""
        if self._answer_in_progress:
            return
        self._answer_in_progress = True
        self.send_button.disable()
        self.start_screen.set_visibility(False)

        add_user_message(self.messages_column, question)
        assistant_message = AssistantMessage(self.messages_column)
        self._scroll_to_bottom()

        try:
            user_session = await self.lazy_user_session.get()
            text_so_far = ""
            async for update in user_session.answer_streamer.stream_answer(question):
                if update.type == "text":
                    text_so_far += update.text
                    assistant_message.show_text(text_so_far)
                elif update.answer is not None:
                    # The final text carries the citation markers, which are only
                    # known once the whole answer is written.
                    assistant_message.finish(update.answer)
                self._scroll_to_bottom()
        except Exception as error:
            logger.error("Answer failed", error=str(error))
            assistant_message.show_error(ANSWER_FAILED_MESSAGE)
        finally:
            self._answer_in_progress = False
            self.send_button.enable()

    @staticmethod
    def _scroll_to_bottom() -> None:
        ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")
