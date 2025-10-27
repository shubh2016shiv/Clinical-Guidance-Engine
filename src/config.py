"""
Configuration settings for the Drug Recommendation Chatbot.

Simple configuration using Pydantic BaseSettings to read from .env file.
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # OpenAI API Configuration
    openai_api_key: str = Field(
        default="", description="OpenAI API Key loaded from .env"
    )
    openai_model_name: str = Field(
        default="gpt-4",
        description="OpenAI model name, can be overridden in .env",
    )

    # OpenAI Model Parameters
    openai_temperature: float = Field(
        default=0.1, description="Creativity vs consistency (0.0-2.0)"
    )
    openai_top_p: float = Field(
        default=0.9, description="Nucleus sampling parameter (0.0-1.0)"
    )
    openai_max_output_tokens: int = Field(
        default=2000, description="Maximum response length"
    )

    # Gemini API Configuration
    gemini_api_key: str = Field(
        default="", description="Google Gemini API Key loaded from .env"
    )
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001", description="Gemini embedding model name"
    )
    gemini_embedding_dimension: int = Field(
        default=1536, description="Dimension size for Gemini embeddings"
    )

    # Application Configuration
    app_name: str = "Drug Recommendation Chatbot"
    app_version: str = "1.0.0"
    debug_mode: bool = False

    # Response Mode Configuration
    enable_streaming: bool = False  # Global streaming mode control
    enable_cleanup: bool = False  # Control resource cleanup at shutdown

    # Vector store configuration
    vector_store_ttl: int = 3600  # 1 hour in seconds

    # File paths configuration
    clinical_guidelines_directory: str = (
        "clinical_guidelines"  # Path relative to project root
    )

    class Config:
        """Pydantic configuration."""

        # Get the project root directory (parent of src directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_file = os.path.join(project_root, ".env")
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
