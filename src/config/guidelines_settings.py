"""Clinical guideline file location and vector store settings.

Big-picture flow:
1. Describe where on disk the clinical guideline documents live.
2. Describe how long OpenAI should keep the hosted file-search store built
   from those documents before it expires.
"""

from pydantic import BaseModel, Field

# See paths.py for what Pydantic's BaseModel and Field do.


# GuidelinesSettings: groups the settings the clinical guideline agent
# needs to find its source documents and manage the OpenAI-hosted store
# built from them (see src/bootstrap/prepare_guidelines_store.py).
class GuidelinesSettings(BaseModel):
    """Fields used to locate and manage hosted guideline files."""

    vector_store_ttl: int = Field(
        default=3600,
        ge=0,
        description="Vector store TTL in seconds (3600 = 1 hour). 0 = no expiry.",
    )
    # vector_store_ttl: "TTL" (time to live) is how long OpenAI keeps the
    # hosted guideline store before deleting it automatically. 0 means
    # "never expire".

    clinical_guidelines_directory: str = Field(
        default="clinical_guidelines",
        description="Path to the clinical guidelines directory, relative to the project root.",
    )
    # clinical_guidelines_directory: the folder containing the guideline
    # documents to upload. If given as a relative path (like the default),
    # it is resolved against PROJECT_ROOT elsewhere (see
    # src/bootstrap/check_settings.py).
