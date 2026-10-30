"""The guideline files panel in the sidebar, kept separate from the chat."""

from nicegui import events, ui

from src.clinical_guideline_agent import IndexedGuidelineFile
from src.logs import get_component_logger

from ..services.user_session import LazyUserSession
from ..web_ui_settings import GUIDELINE_UPLOAD_ACCEPT, GUIDELINE_UPLOAD_MAX_FILE_SIZE_MB

logger = get_component_logger("HealthcareAssistantWeb")

MEGABYTE = 1024 * 1024


def format_file_size(size_bytes: int) -> str:
    """Return a size a person can read, e.g. 2.4 MB."""
    if size_bytes <= 0:
        return ""
    if size_bytes < MEGABYTE:
        return f"{max(1, round(size_bytes / 1024))} KB"
    return f"{size_bytes / MEGABYTE:.1f} MB"


class GuidelinesPanel:
    """Lists the indexed guideline files, and lets the user add or delete them."""

    def __init__(self, lazy_user_session: LazyUserSession) -> None:
        self.lazy_user_session = lazy_user_session

    def build(self) -> None:
        """Draw the panel. Call this inside the sidebar."""
        with ui.row().classes("w-full items-center justify-between no-wrap"):
            with ui.row().classes("items-center gap-1 no-wrap"):
                ui.label("Guideline Files").classes("sidebar-heading")
                ui.icon("help_outline", size="0.9rem").style("color: var(--text-faint)").tooltip(
                    "Uploaded files are indexed and cited in answers."
                )
            ui.button("Upload", icon="file_upload", on_click=self._open_file_picker).props(
                "flat dense no-caps color=primary"
            )

        self._build_dropzone()

        # Shown only while files are being indexed.
        with ui.row().classes("items-center gap-2 mt-1") as self.indexing_notice:
            ui.spinner(size="0.9rem", color="primary")
            self.indexing_label = ui.label().classes("text-xs muted-text")
        self.indexing_notice.set_visibility(False)

        self.file_list = ui.column().classes("w-full gap-2 mt-1")
        with self.file_list:
            ui.label("Loading...").classes("file-meta")

        # Fill the list once the page is up; this waits for the connections to be ready.
        ui.timer(0.1, self.refresh, once=True)

    def _build_dropzone(self) -> None:
        """Draw the drag-and-drop box, with the real uploader hidden underneath it."""
        with ui.element("div").classes("dropzone w-full"):
            self.upload = (
                ui.upload(
                    multiple=True,
                    auto_upload=True,
                    max_file_size=GUIDELINE_UPLOAD_MAX_FILE_SIZE_MB * MEGABYTE,
                    on_multi_upload=self._index_uploaded_files,
                    on_rejected=lambda: ui.notify(
                        "That file is too large or not a PDF or Markdown file.", type="warning"
                    ),
                )
                .props(f'accept="{GUIDELINE_UPLOAD_ACCEPT}" flat')
                .classes("w-full")
            )
            with ui.column().classes("dropzone-overlay w-full items-center gap-1"):
                ui.icon("cloud_upload", size="1.8rem").style("color: var(--brand)")
                ui.label("Drag & drop files here").classes("dropzone-title")
                ui.label(f"PDF or Markdown (Max {GUIDELINE_UPLOAD_MAX_FILE_SIZE_MB}MB)").classes(
                    "dropzone-hint"
                )
                ui.label("or Browse Files").classes("browse-link").on(
                    "click", self._open_file_picker
                )

    def _open_file_picker(self) -> None:
        """Open the browser's file chooser for the hidden uploader."""
        self.upload.run_method("pickFiles")

    async def refresh(self) -> None:
        """Redraw the file list from what the vector store really holds."""
        try:
            user_session = await self.lazy_user_session.get()
            indexed_files = await user_session.guideline_library.list_files()
        except Exception as error:
            logger.error("Could not list guideline files", error=str(error))
            self.file_list.clear()
            with self.file_list:
                ui.label("Could not load the guideline files.").classes("text-xs text-negative")
            return

        self.file_list.clear()
        with self.file_list:
            if not indexed_files:
                ui.label("No guidelines added yet.").classes("file-meta")
            for indexed_file in indexed_files:
                self._build_file_card(indexed_file)

    def _build_file_card(self, indexed_file: IndexedGuidelineFile) -> None:
        is_markdown = indexed_file.filename.lower().endswith(".md")
        is_ready = indexed_file.status == "completed"

        with ui.row().classes("file-card w-full items-center no-wrap gap-2"):
            ui.label("MD" if is_markdown else "PDF").classes(
                "file-badge is-markdown" if is_markdown else "file-badge"
            )

            with ui.column().classes("grow gap-0 min-w-0"):
                ui.label(indexed_file.filename).classes("file-name truncate w-full").tooltip(
                    indexed_file.filename
                )
                size_text = format_file_size(indexed_file.size_bytes)
                state_text = "Indexed" if is_ready else indexed_file.status.replace("_", " ")
                ui.label(" • ".join(part for part in (size_text, state_text) if part)).classes(
                    "file-meta"
                )

            if is_ready:
                ui.icon("check_circle", size="1.1rem").style("color: var(--q-positive)")
            else:
                ui.spinner(size="0.9rem", color="primary")

            with ui.button(icon="more_horiz").props("flat round dense color=grey-6"):
                with (
                    ui.menu(),
                    ui.menu_item(
                        on_click=lambda _, file=indexed_file: self._confirm_and_delete(file)
                    ),
                ):
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        ui.icon("delete_outline", size="1.1rem").style("color: var(--q-negative)")
                        ui.label("Remove")

    async def _index_uploaded_files(self, upload_event: events.MultiUploadEventArguments) -> None:
        file_count = len(upload_event.files)
        self.indexing_label.set_text(f"Indexing {file_count} file(s)...")
        self.indexing_notice.set_visibility(True)
        try:
            user_session = await self.lazy_user_session.get()
            result = await user_session.guideline_library.upload_files(upload_event.files)
            if result.indexed_names:
                ui.notify(f"Indexed {len(result.indexed_names)} file(s).", type="positive")
            if result.skipped_names:
                ui.notify(
                    f"Skipped (not PDF or Markdown): {', '.join(result.skipped_names)}",
                    type="warning",
                )
        except Exception as error:
            logger.error("Guideline upload failed", error=str(error))
            ui.notify("Could not index the files. Please try again.", type="negative")
        finally:
            self.indexing_notice.set_visibility(False)
            self.upload.reset()  # clear the finished files from the upload box
        await self.refresh()

    async def _confirm_and_delete(self, indexed_file: IndexedGuidelineFile) -> None:
        with ui.dialog() as confirm_dialog, ui.card().classes("gap-2"):
            ui.label(f'Remove "{indexed_file.filename}"?').classes("font-semibold")
            ui.label("Answers will no longer use it. This cannot be undone.").classes(
                "muted-text text-sm"
            )
            with ui.row().classes("justify-end w-full gap-2"):
                ui.button("Cancel", on_click=lambda: confirm_dialog.submit(False)).props(
                    "flat no-caps"
                )
                ui.button("Remove", on_click=lambda: confirm_dialog.submit(True)).props(
                    "unelevated no-caps color=negative"
                )
        if not await confirm_dialog:
            return

        try:
            user_session = await self.lazy_user_session.get()
            await user_session.guideline_library.delete_file(indexed_file.file_id)
            ui.notify(f'Removed "{indexed_file.filename}".', type="positive")
        except Exception as error:
            logger.error("Could not remove guideline file", error=str(error))
            ui.notify("Could not remove the file. Please try again.", type="negative")
        await self.refresh()
