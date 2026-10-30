"""Colours and fonts, in one place so the whole interface changes together."""

from pathlib import Path
from typing import Final

PRIMARY_COLOR: Final[str] = "#006BFF"  # the blue used for buttons and links
SECONDARY_COLOR: Final[str] = "#6550EF"  # the violet accent in the light theme
POSITIVE_COLOR: Final[str] = "#16A34A"
WARNING_COLOR: Final[str] = "#D97706"
NEGATIVE_COLOR: Final[str] = "#DC2626"

FONT_STYLESHEET_URL: Final[str] = (
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
)
STYLESHEET_PATH: Final[Path] = Path(__file__).with_name("healthcare_theme.css")
