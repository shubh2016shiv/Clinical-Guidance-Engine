"""
Configuration settings for the Chainlit UI application.

Defines UI-specific settings including conversation persistence, display options,
and application metadata.
"""

from pathlib import Path
from typing import Final

# Application Metadata
APP_TITLE: Final[str] = "Healthcare Assistant"
APP_DESCRIPTION: Final[str] = "Answers grounded in the available clinical guideline documents"
APP_VERSION: Final[str] = "1.0.0"

# Conversation Persistence Settings
CHAT_HISTORY_ENABLED: Final[bool] = True
MAX_HISTORY_MESSAGES: Final[int] = 50
CONVERSATION_DIR: Final[Path] = Path("healthcare_assistant_ui/conversations")

# Ensure conversation directory exists
CONVERSATION_DIR.mkdir(parents=True, exist_ok=True)

# UI Display Settings
SHOW_INLINE_CITATIONS: Final[bool] = True
SHOW_CITATION_ELEMENTS: Final[bool] = True
STREAM_RESPONSES: Final[bool] = True

# Session Configuration
SESSION_TIMEOUT_MINUTES: Final[int] = 30

# Welcome Message
WELCOME_MESSAGE: Final[str] = (
    f"Hello! I'm {APP_TITLE}. Ask me about the available clinical guidelines. "
    "I'll cite the guideline files when they support an answer."
)

# Error Messages
ERROR_MESSAGE_INITIALIZATION: Final[str] = (
    "I'm having trouble initializing. Please refresh and try again."
)
ERROR_MESSAGE_PROCESSING: Final[str] = (
    "I encountered an error processing your request. Please try again."
)
ERROR_MESSAGE_AGENT_NOT_READY: Final[str] = "I'm not ready yet. Please wait a moment and try again."
