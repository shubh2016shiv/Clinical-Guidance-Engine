"""Validate settings needed by the chatbot before creating service clients."""

from pathlib import Path

from src.config import PROJECT_ROOT, Settings


def check_settings(
    settings: Settings, *, use_drug_database: bool, use_clinical_guidelines: bool
) -> None:
    """Raise a clear error for missing required settings or guideline files."""
    if not settings.openai_api_key.strip():
        raise ValueError("OPENAI_API_KEY is required")
    if not settings.openai_model_name.strip():
        raise ValueError("OPENAI_MODEL_NAME is required")
    if not settings.redis_host.strip():
        raise ValueError("REDIS_HOST is required for the configured cache")
    if use_drug_database:
        for name in ("milvus_host", "milvus_collection_name", "openai_embedding_model"):
            if not getattr(settings, name).strip():
                raise ValueError(f"{name.upper()} is required for drug search")
        # Pydantic Settings already checks that ports and dimensions are positive.
    if use_clinical_guidelines:
        directory = Path(settings.clinical_guidelines_directory)
        if not directory.is_absolute():
            directory = PROJECT_ROOT / directory
        if not directory.is_dir():
            raise ValueError(f"Clinical guidelines directory does not exist: {directory}")
