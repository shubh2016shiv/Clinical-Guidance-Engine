"""Clinical guideline file location and vector store settings."""

from pydantic import BaseModel, Field


class GuidelinesSettings(BaseModel):
    """Fields used to locate and manage hosted guideline files."""

    vector_store_ttl: int = Field(
        default=3600,
        ge=0,
        description="Vector store TTL in seconds (3600 = 1 hour). 0 = no expiry.",
    )

    clinical_guidelines_directory: str = Field(
        default="clinical_guidelines",
        description="Path to the clinical guidelines directory, relative to the project root.",
    )
