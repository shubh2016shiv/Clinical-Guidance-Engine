"""The friendly doctor robot next to the start-screen greeting."""

from pathlib import Path

from nicegui import ui

_MASCOT_PATH = Path(__file__).resolve().parents[1] / "assets" / "healthcare-robot.png"


def build_mascot() -> None:
    """Load the local transparent mascot without relying on an external image host."""
    ui.image(str(_MASCOT_PATH)).props(
        'fit=contain no-spinner alt="Friendly Healthcare Assistant robot waving"'
    ).classes("greeting-mascot")
