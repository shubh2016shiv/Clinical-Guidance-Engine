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
    # Used by: response_api_agent/managers/*, llm_provider/*, app.py
    openai_api_key: str = Field(
        default="",
        description="OpenAI API Key loaded from .env. Required for OpenAI API access.",
    )
    openai_model_name: str = Field(
        default="gpt-4",
        description="OpenAI model name, can be overridden in .env. Used by response API managers and LLM providers.",
    )

    # OpenAI Model Parameters
    # Used by: response_api_agent/managers/*, llm_provider/*
    openai_temperature: float = Field(
        default=0.1,
        description="Creativity vs consistency (0.0-2.0). Lower = more deterministic. Used by response API managers.",
    )
    openai_top_p: float = Field(
        default=0.9,
        description="Nucleus sampling parameter (0.0-1.0). Used by response API managers.",
    )
    openai_max_output_tokens: int = Field(
        default=2000,
        description="Maximum response length in tokens. Used by response API managers.",
    )

    # Gemini API Configuration
    # Used by: milvus_provider.py, embedding_provider/gemini_embedding_provider.py
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API Key loaded from .env. Required for embedding generation.",
    )
    gemini_embedding_model: str = Field(
        default="text-embedding-004",
        description="Gemini embedding model name. Must match ingestion config (default: text-embedding-004).",
    )
    gemini_embedding_dimension: int = Field(
        default=768,
        description="Dimension size for Gemini embeddings. Must match Milvus collection dimension (default: 768).",
    )

    # Milvus Connection Configuration
    # Used by: milvus_provider.py (connection management - connect_sync, connect methods)
    milvus_host: str = Field(
        default="localhost",
        description="Milvus server host address. Used by milvus_provider.py for connection.",
    )
    milvus_port: int = Field(
        default=19530,
        description="Milvus server port. Default: 19530. Used by milvus_provider.py for connection.",
    )
    milvus_alias: str = Field(
        default="default",
        description="Milvus connection alias. Used by milvus_provider.py for connection management.",
    )
    milvus_timeout: int = Field(
        default=30,
        description="Milvus connection timeout in seconds. Used by milvus_provider.py.",
    )
    milvus_user: str = Field(
        default="",
        description="Milvus authentication username (optional). Used by milvus_provider.py if authentication enabled.",
    )
    milvus_password: str = Field(
        default="",
        description="Milvus authentication password (optional). Can be set in .env file. Used by milvus_provider.py if authentication enabled.",
    )
    milvus_secure: bool = Field(
        default=False,
        description="Enable TLS/SSL for Milvus connection. Used by milvus_provider.py for secure connections.",
    )

    # Milvus Collection Configuration
    # Used by: milvus_provider.py (collection management - _load_collection, get_collection_info methods)
    milvus_collection_name: str = Field(
        default="pharmaceutical_drugs",
        description="Milvus collection name. Must match ingestion config. Used by milvus_provider.py for collection operations.",
    )
    milvus_collection_description: str = Field(
        default="Pharmaceutical drug database with embeddings for LLM lookup",
        description="Milvus collection description. Used by milvus_provider.py.",
    )
    milvus_embedding_dimension: int = Field(
        default=768,
        description="Milvus collection embedding dimension. Must match gemini_embedding_dimension (default: 768). Used by milvus_provider.py for validation.",
    )
    milvus_load_on_startup: bool = Field(
        default=True,
        description="Automatically load collection into memory on connection. Used by milvus_provider.py.",
    )

    # Milvus Search Configuration
    # Used by: milvus_provider.py (search operations - search, search_by_text methods)
    milvus_search_top_k: int = Field(
        default=10,
        description="Default number of search results to return. Used by milvus_provider.py search methods.",
    )
    milvus_search_ef: int = Field(
        default=64,
        description="HNSW index search parameter (ef). Higher = better accuracy, slower search. Used by milvus_provider.py for search operations.",
    )
    milvus_search_output_fields: list[str] = Field(
        default_factory=lambda: [
            "drug_name",
            "drug_class",
            "drug_sub_class",
            "therapeutic_category",
            "route_of_administration",
            "formulation",
            "dosage_strengths",
            "search_text",
        ],
        description="List of fields to return in search results. Used by milvus_provider.py search methods.",
    )

    # Application Configuration
    # Used by: app.py, various managers for application metadata
    app_name: str = "Drug Recommendation Chatbot"
    app_version: str = "1.0.0"
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode for detailed logging. Used throughout the application.",
    )

    # Response Mode Configuration
    # Used by: response_api_agent/managers/stream_manager.py, chat_manager.py
    enable_streaming: bool = Field(
        default=True,
        description="Global streaming mode control. Used by stream_manager.py and chat_manager.py.",
    )
    enable_cleanup: bool = Field(
        default=False,
        description="Control resource cleanup at shutdown. Used by response_api_manager.py.",
    )

    # Vector Store Configuration
    # Used by: response_api_agent/managers/vector_store_manager.py
    vector_store_ttl: int = Field(
        default=3600,
        description="Vector store time-to-live in seconds (1 hour default). Used by vector_store_manager.py for cleanup.",
    )

    # File Paths Configuration
    # Used by: response_api_agent/managers/upload_manager.py, vector_store_manager.py
    clinical_guidelines_directory: str = Field(
        default="clinical_guidelines",
        description="Path to clinical guidelines directory relative to project root. Used by upload_manager.py and vector_store_manager.py.",
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
