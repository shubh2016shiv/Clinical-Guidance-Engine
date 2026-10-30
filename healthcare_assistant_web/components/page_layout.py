"""Puts the sidebar, the header and the chat together into the home page."""

from nicegui import context, ui

from ..services.shared_services import SharedServicesManager
from ..services.user_session import LazyUserSession
from ..styling.theme import (
    FONT_STYLESHEET_URL,
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    STYLESHEET_PATH,
    WARNING_COLOR,
)
from ..web_ui_settings import APP_TAGLINE, APP_TITLE
from .chat_area import ChatArea
from .connection_status import build_connection_status
from .guidelines_panel import GuidelinesPanel


def build_home_page(shared_services_manager: SharedServicesManager) -> None:
    """Build the whole page for one browser tab."""
    # Each tab gets its own conversation; it is created only when first needed.
    lazy_user_session = LazyUserSession(shared_services_manager)
    context.client.on_disconnect(lazy_user_session.close)

    dark_mode = _apply_theme()
    chat_area = ChatArea(lazy_user_session)

    # top_corner and bottom_corner stretch the sidebar over the header and footer
    # rows, so it runs the full height of the page. They also make the question
    # box centre over the chat instead of over the whole window.
    with (
        ui.left_drawer(value=True, bordered=False, top_corner=True, bottom_corner=True)
        .props("width=300")
        .classes("app-sidebar p-4 gap-4")
    ):
        _build_brand()
        ui.button("New Chat", icon="add", on_click=chat_area.start_new_chat).props(
            "unelevated no-caps"
        ).classes("new-chat-button w-full")
        GuidelinesPanel(lazy_user_session).build()
        ui.space()
        _build_promo_card()

    with ui.header().classes("app-header items-center justify-end px-5 py-3 gap-2"):
        build_connection_status(shared_services_manager)
        ui.button(icon="dark_mode", on_click=dark_mode.toggle).props(
            "flat round dense color=grey-7"
        ).tooltip("Switch between light and dark")

    with ui.footer().classes("items-center px-4 pb-3 pt-1"):
        chat_area.build_input_bar()

    with ui.column().classes("w-full items-center"):
        chat_area.build_messages()


def _build_brand() -> None:
    with ui.row().classes("items-center gap-3 no-wrap"):
        with ui.element("div").classes("brand-mark"):
            ui.icon("medical_services", size="1.3rem")
        with ui.column().classes("gap-0 min-w-0"):
            ui.label(APP_TITLE).classes("brand-name")
            ui.label(APP_TAGLINE).classes("brand-tagline")


def _build_promo_card() -> None:
    with ui.column().classes("promo-card w-full gap-1"):
        ui.icon("auto_stories", size="1.4rem").style("color: var(--brand)")
        ui.label("Evidence-based care made simple").classes("text-sm font-semibold")
        ui.label("Ask questions, explore guidelines and get answers you can trace.").classes(
            "text-xs muted-text"
        )


def _apply_theme() -> ui.dark_mode:
    ui.colors(
        primary=PRIMARY_COLOR,
        secondary=SECONDARY_COLOR,
        positive=POSITIVE_COLOR,
        warning=WARNING_COLOR,
        negative=NEGATIVE_COLOR,
    )
    ui.add_head_html(f'<link rel="stylesheet" href="{FONT_STYLESHEET_URL}">')
    ui.add_css(STYLESHEET_PATH.read_text(encoding="utf-8"))
    # Light by default, which is what the design is built for; the header
    # button switches to dark for anyone who prefers it.
    return ui.dark_mode(value=False)
