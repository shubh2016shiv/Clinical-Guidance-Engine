"""Fixed values for the web interface: titles, upload rules and example questions."""

from pathlib import Path
from typing import Final, NamedTuple

APP_TITLE: Final[str] = "Healthcare Assistant"
APP_TAGLINE: Final[str] = "Evidence • Guidelines • Better Care"
APP_SUBTITLE: Final[str] = (
    "Answers grounded in clinical guideline documents and the pharmaceutical drug catalog"
)
GREETING_LINE: Final[str] = "I'm your Healthcare Assistant"
GREETING_HINT: Final[str] = (
    "Ask about clinical guidelines, drug information, or evidence-based recommendations."
)

# Where the server listens.
SERVER_PORT: Final[int] = 8080

# Guideline uploads
GUIDELINE_FILE_SUFFIXES: Final[frozenset[str]] = frozenset({".pdf", ".md"})
GUIDELINE_UPLOAD_ACCEPT: Final[str] = ".pdf,.md"  # the browser's file picker filter
GUIDELINE_UPLOAD_MAX_FILE_SIZE_MB: Final[int] = 100

# Saved conversations, one JSON file per browser tab.
CONVERSATIONS_DIR: Final[Path] = Path(__file__).resolve().parent / "conversations"


class StarterQuestion(NamedTuple):
    """One example question offered on the start screen."""

    icon: str  # a Material icon name, e.g. "lightbulb"
    label: str
    question: str


STARTER_QUESTIONS: Final[list[StarterQuestion]] = [
    StarterQuestion(
        "lightbulb",
        "Summarise a guideline",
        "Summarise the key recommendations in the guidelines I uploaded.",
    ),
    StarterQuestion(
        "medication",
        "Drug dosing",
        "What do the guidelines recommend as first-line drug treatment, and at what doses?",
    ),
    StarterQuestion(
        "warning_amber",
        "Adverse effects",
        "What adverse effects should I watch for with these medications?",
    ),
    StarterQuestion(
        "compare_arrows",
        "Compare treatments",
        "Compare the treatment options the guidelines recommend.",
    ),
]
