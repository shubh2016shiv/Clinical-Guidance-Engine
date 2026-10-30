"""A small light in the header showing whether the drug catalog is connected."""

from nicegui import ui

from ..services.shared_services import SharedServicesManager

STATUS_TEXT = {
    "connecting": "Connecting",
    "ready": "Ready",
    "failed": "Offline",
}
STATUS_COLOR = {
    "connecting": "warning",
    "ready": "positive",
    "failed": "negative",
}


def build_connection_status(shared_services_manager: SharedServicesManager) -> None:
    """Draw the status pill and keep it up to date while the page is open."""
    with ui.row().classes("status-pill items-center gap-2 no-wrap"):
        light = ui.icon("circle", size="0.6rem")
        text = ui.label().classes("text-sm font-medium")

    def update_status() -> None:
        status = shared_services_manager.status
        light.props(f"color={STATUS_COLOR[status]}")
        text.set_text(STATUS_TEXT[status])
        if status == "failed":
            text.tooltip(shared_services_manager.last_error)

    update_status()
    ui.timer(1.0, update_status)
