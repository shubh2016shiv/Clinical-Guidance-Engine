"""
Configuration settings for the Drug Recommendation Chatbot.

Simple configuration using Pydantic BaseSettings to read from .env file.
"""

from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # OpenAI API Configuration
    openai_api_key: str = ""

    openai_model_name: str = "gpt-4o-mini"

    # Application Configuration
    app_name: str = "Drug Recommendation Chatbot"
    app_version: str = "1.0.0"
    debug_mode: bool = False

    # Vector store configuration
    vector_store_ttl: int = 3600  # 1 hour in seconds

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings