"""What the page shows before the first question: a greeting and some example questions."""

from collections.abc import Awaitable, Callable

from nicegui import ui

from ..web_ui_settings import APP_TITLE, GREETING_HINT, GREETING_LINE, STARTER_QUESTIONS
from .mascot import build_mascot

_STARTER_SYMBOLS = {
    "lightbulb": "💡",
    "medication": "💊",
    "warning_amber": "🩺",
    "compare_arrows": "📄",
}


def build_start_screen(on_question_picked: Callable[[str], Awaitable[None]]) -> ui.element:
    """Draw the greeting and return it, so the chat can hide it after the first question."""
    with ui.row().classes("greeting-section items-center no-wrap") as start_screen:
        build_mascot()

        with ui.column().classes("greeting-content grow min-w-0"):
            with ui.row().classes("items-center gap-2 no-wrap"):
                ui.label("Hello!").classes("greeting-title")
                ui.icon("auto_awesome", size="1.4rem").classes("greeting-sparkle")
            ui.label(GREETING_LINE).classes("greeting-subtitle")
            ui.label(GREETING_HINT).classes("greeting-hint")

            with ui.row().classes("starter-actions gap-2 flex-wrap"):
                for starter in STARTER_QUESTIONS:
                    # "question=..." fixes this chip's question now; without it every
                    # chip would end up using the last question in the loop.
                    with (
                        ui.row()
                        .classes("starter-chip items-center gap-2 no-wrap")
                        .on(
                            "click",
                            lambda _, question=starter.question: on_question_picked(question),
                        )
                    ):
                        ui.label(_STARTER_SYMBOLS[starter.icon]).classes("starter-symbol")
                        ui.label(
                            "Summarize guideline" if starter.icon == "lightbulb" else starter.label
                        )

    ui.label(APP_TITLE).classes("hidden")  # keeps the page title searchable
    return start_screen
