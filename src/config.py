"""
Configuration settings for the Drug Recommendation Chatbot (Asclepius).

Enterprise-ready configuration using Pydantic v2 Settings with:
- Strong typing, validation, and cross-field consistency checks
- Frozen (immutable) settings after load to prevent accidental mutation
- Structured logging of loaded configuration (with secret masking)
- Thread-safe singleton access via functools.lru_cache
- Rejects unknown environment variables (extra = "forbid") to catch typos
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Project root resolution (used for .env file path and defaults)
# ---------------------------------------------------------------------------


def _resolve_project_root() -> Path:
    """Resolve the project root directory (parent of the ``src`` directory)."""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT: Path = _resolve_project_root()
ENV_FILE: Path = PROJECT_ROOT / ".env"


# ---------------------------------------------------------------------------
# Settings class
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Application settings loaded from ``.env`` file and environment variables.

    All settings are **frozen** after initialisation — use ``model_copy()`` if
    you need a mutated copy for testing or overrides.

    Field names follow the ``lowercase_underscore`` convention; environment
    variable names are matched **case-insensitively** (the pydantic-settings
    default).
    """

    # ------------------------------------------------------------------
    # Pydantic model configuration
    # ------------------------------------------------------------------

    model_config = SettingsConfigDict(
        # -- .env file ---------------------------------------------------
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        # -- behaviour ---------------------------------------------------
        case_sensitive=False,  # OPENAI_API_KEY → openai_api_key
        extra="forbid",  # reject unknown env vars (typo guard)
        frozen=True,  # settings are immutable after load
        validate_default=True,  # run validators even on defaults
        # -- JSON parsing for complex types ------------------------------
        env_parse_none_str="null",
    )

    # ------------------------------------------------------------------
    # OpenAI API Configuration
    # ------------------------------------------------------------------

    openai_api_key: str = Field(
        default="",
        description="OpenAI API Key. Required for OpenAI API access.",
    )
    openai_model_name: str = Field(
        default="gpt-4o",
        description="OpenAI model name (e.g. gpt-4o, gpt-4o-mini).",
    )
    openai_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic, 2.0 = max creativity).",
    )
    openai_top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter. Some Responses API models may not support this.",
    )
    openai_max_output_tokens: int = Field(
        default=2000,
        ge=1,
        description="Maximum response length in tokens.",
    )

    # ------------------------------------------------------------------
    # Google Gemini API Configuration (Embedding Provider)
    # ------------------------------------------------------------------

    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API Key. Required for embedding generation.",
    )
    gemini_embedding_model: str = Field(
        default="text-embedding-004",
        description="Gemini embedding model name. Must match ingestion config.",
    )
    gemini_embedding_dimension: int = Field(
        default=768,
        ge=1,
        description="Dimension size for Gemini embeddings. Must match Milvus collection dimension.",
    )

    # ------------------------------------------------------------------
    # Default LLM Provider
    # ------------------------------------------------------------------

    default_llm_provider: str = Field(
        default="openai",
        description="Default LLM provider slug (e.g. 'openai'). Used by the LLM provider factory.",
    )
    default_model_name: str = Field(
        default="gpt-4o",
        description="Default LLM model name used when no explicit model is specified.",
    )

    # ------------------------------------------------------------------
    # Default Embedding Provider
    # ------------------------------------------------------------------

    default_embedding_provider: str = Field(
        default="gemini",
        description="Default embedding provider (gemini or openai).",
    )
    default_embedding_model: str = Field(
        default="text-embedding-004",
        description="Default embedding model name used when no explicit model is specified.",
    )

    # ------------------------------------------------------------------
    # Milvus — Connection
    # ------------------------------------------------------------------

    milvus_host: str = Field(
        default="localhost",
        description="Milvus server host. Use 'milvus-standalone' inside Docker Compose network.",
    )
    milvus_port: int = Field(
        default=19530,
        ge=1,
        le=65535,
        description="Milvus gRPC API port.",
    )
    milvus_alias: str = Field(
        default="default",
        description="Milvus connection alias.",
    )
    milvus_timeout: int = Field(
        default=30,
        ge=1,
        description="Milvus connection timeout in seconds.",
    )
    milvus_user: str = Field(
        default="",
        description="Milvus authentication username (optional).",
    )
    milvus_password: str = Field(
        default="",
        description="Milvus authentication password (optional).",
    )
    milvus_secure: bool = Field(
        default=False,
        description="Enable TLS/SSL for Milvus connection.",
    )

    # ------------------------------------------------------------------
    # Milvus — Collection
    # ------------------------------------------------------------------

    milvus_collection_name: str = Field(
        default="pharmaceutical_drugs",
        description="Milvus collection name. Must match ingestion config.",
    )
    milvus_collection_description: str = Field(
        default="Pharmaceutical drug database with embeddings for LLM lookup",
        description="Human-readable collection description.",
    )
    milvus_embedding_dimension: int = Field(
        default=768,
        ge=1,
        description="Milvus collection embedding dimension. Must equal gemini_embedding_dimension.",
    )
    milvus_load_on_startup: bool = Field(
        default=True,
        description="Load collection into memory automatically on connection.",
    )

    # ------------------------------------------------------------------
    # Milvus — Search
    # ------------------------------------------------------------------

    milvus_search_top_k: int = Field(
        default=10,
        ge=1,
        description="Default number of nearest-neighbour results to return.",
    )
    milvus_search_ef: int = Field(
        default=64,
        ge=1,
        description="HNSW index search parameter (ef). Higher = better recall, slower search.",
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
        description="List of scalar fields to return with each search result. "
        'Set via env as a JSON array, e.g. \'["drug_name","drug_class"]\'.',
    )

    # ------------------------------------------------------------------
    # Redis Cache
    # ------------------------------------------------------------------

    redis_host: str = Field(
        default="localhost",
        description="Redis server host. Use 'redis' inside Docker Compose network.",
    )
    redis_port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis server port.",
    )
    redis_db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database index (0–15).",
    )
    redis_password: str = Field(
        default="redis123",
        description="Redis authentication password.",
    )
    redis_max_connections: int = Field(
        default=50,
        ge=1,
        description="Maximum connections in the Redis connection pool.",
    )
    redis_socket_timeout: int = Field(
        default=5,
        ge=1,
        description="Redis socket timeout in seconds.",
    )
    redis_key_prefix: str = Field(
        default="Asclepius_Conversations",
        description="Prefix for all Redis keys (namespace isolation).",
    )
    redis_default_session_ttl: int = Field(
        default=7200,
        ge=0,
        description="Default TTL for session data in seconds (7200 = 2 hours).",
    )
    redis_default_history_ttl: int = Field(
        default=1800,
        ge=0,
        description="Default TTL for chat-history cache in seconds (1800 = 30 minutes).",
    )

    # -- Redis resilience (circuit breaker + retry) -------------------
    redis_circuit_breaker_fail_max: int = Field(
        default=5,
        ge=1,
        description="Number of consecutive failures before the Redis circuit breaker opens.",
    )
    redis_circuit_breaker_timeout: int = Field(
        default=30,
        ge=1,
        description="Seconds the Redis circuit breaker stays open before attempting half-open recovery.",
    )
    redis_retry_max_attempts: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts for transient Redis operation failures. 0 = no retries.",
    )
    redis_retry_base_delay: float = Field(
        default=0.1,
        ge=0.0,
        description="Base delay in seconds for exponential backoff between Redis retries.",
    )
    redis_retry_max_delay: float = Field(
        default=2.0,
        ge=0.0,
        description="Maximum delay in seconds for exponential backoff between Redis retries.",
    )

    # ------------------------------------------------------------------
    # MongoDB (Persistent Operational Data)
    # ------------------------------------------------------------------

    mongo_host: str = Field(
        default="localhost",
        description="MongoDB server host. Use 'mongodb' inside Docker Compose network.",
    )
    mongo_port: int = Field(
        default=27017,
        ge=1,
        le=65535,
        description="MongoDB server port.",
    )
    mongo_root_user: str = Field(
        default="admin",
        description="MongoDB root username.",
    )
    mongo_root_password: str = Field(
        default="password123",
        description="MongoDB root password.",
    )
    mongo_db_name: str = Field(
        default="asclepius",
        description="MongoDB database name for the application.",
    )
    mongo_auth_source: str = Field(
        default="admin",
        description="MongoDB authentication source database.",
    )

    # ------------------------------------------------------------------
    # MinIO S3-Compatible Storage (Milvus Object Backend)
    # ------------------------------------------------------------------

    minio_host: str = Field(
        default="localhost",
        description="MinIO server host. Use 'minio' inside Docker Compose network.",
    )
    minio_api_port: int = Field(
        default=9000,
        ge=1,
        le=65535,
        description="MinIO S3-compatible API port.",
    )
    minio_console_port: int = Field(
        default=9001,
        ge=1,
        le=65535,
        description="MinIO web console port (browser UI).",
    )
    minio_root_user: str = Field(
        default="minioadmin",
        description="MinIO root username / access key.",
    )
    minio_root_password: str = Field(
        default="minioadmin",
        description="MinIO root password / secret key.",
    )
    minio_bucket_name: str = Field(
        default="milvus-bucket",
        description="Default MinIO bucket name for Milvus storage.",
    )
    minio_secure: bool = Field(
        default=False,
        description="Enable TLS/SSL for MinIO connections.",
    )

    # ------------------------------------------------------------------
    # Application Metadata & Behaviour
    # ------------------------------------------------------------------

    app_name: str = Field(
        default="Drug Recommendation Chatbot",
        description="Human-readable application name.",
    )
    app_version: str = Field(
        default="1.0.0",
        description="Application version (semver).",
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode (verbose logging, dev helpers).",
    )
    enable_streaming: bool = Field(
        default=True,
        description="Global toggle for streaming (SSE) responses.",
    )
    enable_cleanup: bool = Field(
        default=False,
        description="Enable resource cleanup on agent shutdown.",
    )

    # ------------------------------------------------------------------
    # Vector Store
    # ------------------------------------------------------------------

    vector_store_ttl: int = Field(
        default=3600,
        ge=0,
        description="Vector store TTL in seconds (3600 = 1 hour). 0 = no expiry.",
    )
    clinical_guidelines_directory: str = Field(
        default="clinical_guidelines",
        description="Path to the clinical guidelines directory, relative to the project root.",
    )

    # ------------------------------------------------------------------
    # Docker / Infrastructure
    # ------------------------------------------------------------------

    docker_volume_directory: str = Field(
        default=".",
        description="Base directory for Docker Compose persistent volumes. "
        "Defaults to the project root ('.').",
    )

    # ==================================================================
    # Cross-Field Validators
    # ==================================================================

    @model_validator(mode="after")
    def _validate_embedding_dimension_consistency(self) -> Settings:
        """Ensure Gemini and Milvus embedding dimensions agree."""
        if self.gemini_embedding_dimension != self.milvus_embedding_dimension:
            raise ValueError(
                f"gemini_embedding_dimension ({self.gemini_embedding_dimension}) "
                f"must equal milvus_embedding_dimension ({self.milvus_embedding_dimension}). "
                "Mismatched dimensions will cause embedding insertion failures."
            )
        return self

    @model_validator(mode="after")
    def _validate_openai_key_when_openai_embedding(self) -> Settings:
        """Warn (not error) when default embedding provider is OpenAI but no key is set."""
        # This is a soft check — only a warning is emitted, since the user
        # might be providing the key at runtime via a provider config object.
        if self.default_embedding_provider == "openai" and not self.openai_api_key:
            _emit_warning(
                "default_embedding_provider is 'openai' but openai_api_key is empty. "
                "OpenAI embedding calls will fail unless a key is supplied at runtime."
            )
        return self

    @model_validator(mode="after")
    def _validate_provider_names(self) -> Settings:
        """Reject unknown embedding provider names early."""
        valid = {"openai", "gemini"}
        if self.default_embedding_provider not in valid:
            raise ValueError(
                f"default_embedding_provider must be one of {valid}, "
                f"got '{self.default_embedding_provider}'."
            )
        return self

    # ==================================================================
    # Debugging helpers
    # ==================================================================

    # Fields whose values should be masked when logging.
    _SECRET_FIELDS: ClassVar[set[str]] = {
        "openai_api_key",
        "gemini_api_key",
        "milvus_password",
        "redis_password",
        "mongo_root_password",
        "minio_root_password",
    }

    def log_loaded_settings(self, *, logger: Any = None) -> None:
        """Log every loaded setting at DEBUG level, masking secrets.

        Args:
            logger: A structlog-style bound logger.  If *None*, a bare
                ``print`` is used (safe fallback when logging is not yet
                configured).
        """
        log_fn = logger.debug if logger is not None else lambda msg, **kw: print(msg)

        field_names = sorted(self.model_fields.keys())
        for name in field_names:
            raw_value = getattr(self, name)
            if name in self._SECRET_FIELDS:
                display = _mask_secret(str(raw_value))
            else:
                display = raw_value
            log_fn(
                "Settings loaded",
                component="Config",
                subcomponent="Settings",
                field=name,
                value=display,
            )


# ======================================================================
# Module-level utilities
# ======================================================================


def _mask_secret(value: str) -> str:
    """Return a masked version of *value* suitable for logging."""
    if not value:
        return "<not set>"
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "****" + value[-4:]


def _emit_warning(message: str) -> None:
    """Emit a user-visible warning to stderr (safe before logging is up)."""
    print(f"[WARNING ] {message}", file=sys.stderr)


def _check_env_file_exists() -> None:
    """Warn if the ``.env`` file cannot be found."""
    if not ENV_FILE.is_file():
        _emit_warning(
            f"Environment file not found at {ENV_FILE}. "
            "Copy env_example.txt to .env and fill in your API keys, "
            "or export the required variables in your shell."
        )


# ----------------------------------------------------------------------
# Singleton accessor (thread-safe via lru_cache)
# ----------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the frozen, validated global :class:`Settings` singleton.

    The instance is cached (``lru_cache``), so repeated calls are cheap and
    return the **exact same object**.

    On first call:
    1. Warns if ``.env`` is missing.
    2. Instantiates and validates :class:`Settings`.
    3. Logs all settings at DEBUG level (if logging is wired up).
    """
    _check_env_file_exists()

    try:
        settings = Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        # Pretty-print validation errors before re-raising
        _emit_warning(f"Settings validation failed ({exc.error_count()} error(s)):")
        for error in exc.errors():
            _emit_warning(f"  - {error['loc']}: {error['msg']}")
        raise

    # Log loaded settings at DEBUG level (best-effort; logging might not
    # be configured yet at import time, hence the try/except).
    try:
        from src.logs import get_component_logger

        logger = get_component_logger("Config")
        settings.log_loaded_settings(logger=logger)
    except Exception:
        # Logging unavailable — fall back to stderr for key values
        if settings.debug_mode:
            settings.log_loaded_settings(logger=None)

    return settings


# ----------------------------------------------------------------------
# Re-export commonly needed utilities so consumers can do:
#   from src.config import get_settings, PROJECT_ROOT, ENV_FILE
# ----------------------------------------------------------------------

__all__ = [
    "Settings",
    "get_settings",
    "PROJECT_ROOT",
    "ENV_FILE",
]
