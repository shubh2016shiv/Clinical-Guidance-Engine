"""The list of guideline files an answer was based on."""

from nicegui import ui

from src.clinical_guideline_agent.schema import FileCitation


def build_sources_block(sources: list[FileCitation]) -> None:
    """Draw the numbered source list. The numbers match the [1], [2] in the answer."""
    if not sources:
        return

    with ui.column().classes("sources-list w-full gap-1 mt-1"):
        with ui.row().classes("items-center gap-1"):
            ui.icon("menu_book", size="1rem").style("color: var(--brand)")
            ui.label("Sources").classes("text-sm font-semibold")

        for source_number, source in enumerate(sources, start=1):
            with ui.row().classes("items-start gap-2 no-wrap"):
                ui.label(f"[{source_number}]").classes("source-number text-sm")
                ui.label(source.filename).classes("source-row")
