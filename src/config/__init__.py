"""Application configuration grouped by infrastructure service.

Beginner note: this file is this package's "__init__.py", so whatever it
imports becomes available directly as ``from src.config import Settings``
instead of the longer ``from src.config.settings import Settings``.
"""

from .paths import ENV_FILE, PROJECT_ROOT
from .settings import Settings, get_settings

# __all__ lists the names meant for other modules to import from here.
__all__ = ["Settings", "get_settings", "PROJECT_ROOT", "ENV_FILE"]
