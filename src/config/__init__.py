"""Application configuration grouped by infrastructure service."""

from .paths import ENV_FILE, PROJECT_ROOT
from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings", "PROJECT_ROOT", "ENV_FILE"]
