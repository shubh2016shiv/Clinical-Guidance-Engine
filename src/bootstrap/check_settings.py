"""Validate settings needed by the chatbot before creating service clients.

Big-picture flow:
1. Read the already-loaded app settings (see src/config.py).
2. Check that every setting required for the features being used is present.
3. Raise a clear, specific error the moment something is missing, instead of
   letting a confusing failure happen later inside Redis, Milvus, or OpenAI
   client code.

Beginner note: this file deliberately does no "connecting" to anything. It
only reads values already in memory, so it runs instantly and fails fast,
before the app spends time opening real network connections.
"""

from pathlib import Path

# Path is Python's standard way to represent a file or folder location that
# works correctly on both Windows and Linux/Mac, instead of manually joining
# strings with "/" or "\\".
from src.config import PROJECT_ROOT, Settings


# check_settings: the single gatekeeper function that decides whether the
# app has enough configuration to start. Needed so every other startup step
# (connecting to Redis, Milvus, OpenAI) can assume the settings it needs are
# already valid, instead of re-checking them itself.
def check_settings(
    settings: Settings, *, use_drug_database: bool, use_clinical_guidelines: bool
) -> None:
    """Raise a clear error for missing required settings or guideline files."""
    # The "*" below makes "use_drug_database" and "use_clinical_guidelines"
    # keyword-only - callers must write them by name, which makes call
    # sites like "check_settings(settings, use_drug_database=True, ...)"
    # self-explanatory instead of relying on argument order.

    # Step 1: these settings are always required, no matter which optional
    # features are turned on, because every request needs OpenAI and Redis.
    if not settings.openai_api_key.strip():
        # ".strip()" removes surrounding whitespace, so a value that is
        # just blank spaces is treated the same as an empty value.
        raise ValueError("OPENAI_API_KEY is required")
    if not settings.openai_model_name.strip():
        raise ValueError("OPENAI_MODEL_NAME is required")
    if not settings.redis_host.strip():
        raise ValueError("REDIS_HOST is required for the configured cache")

    if use_drug_database:
        # Step 2: only check these settings if the drug-search feature is
        # actually being used - a deployment without it should not be
        # forced to configure Milvus.
        for name in ("milvus_host", "milvus_collection_name", "openai_embedding_model"):
            if not getattr(settings, name).strip():
                # "getattr(settings, name)" reads the attribute whose NAME
                # is the string "name" (e.g. "milvus_host") - this lets the
                # same check be reused for several settings without
                # repeating the same three "if" blocks.
                raise ValueError(f"{name.upper()} is required for drug search")
        # Pydantic Settings already checks that ports and dimensions are positive.

    if use_clinical_guidelines:
        # Step 3: only check the guidelines folder if that feature is
        # actually being used.
        directory = Path(settings.clinical_guidelines_directory)
        # directory: the configured guidelines folder, which might be given
        # as a relative or an absolute path.
        if not directory.is_absolute():
            # Step 4: if it was given as a relative path (like
            # "clinical_guidelines"), resolve it against the project's root
            # folder so it works no matter where the app was started from.
            directory = PROJECT_ROOT / directory
        if not directory.is_dir():
            raise ValueError(f"Clinical guidelines directory does not exist: {directory}")
