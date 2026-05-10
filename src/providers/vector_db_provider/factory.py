"""
Vector Database Provider Factory

This module provides factory functions for creating and managing vector database
provider instances. It supports:
- Multiple provider types (Milvus, future Pinecone, etc.)
- Configuration-driven instantiation (defaults from ``Settings``)
- Provider registry management
- Caching of frequently used providers
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.config import get_settings
from src.logs import get_logger
from src.providers.embedding_provider import (
    EmbeddingProvider,
    EmbeddingTaskType,
    create_embedding_provider,
)
from src.providers.vector_db_provider.exceptions import VectorDBConfigError

from .base import (
    VectorDBCollectionConfig,
    VectorDBConfig,
    VectorDBConnectionConfig,
    VectorDBProvider,
    VectorDBSearchConfig,
)
from .milvus_provider import MilvusProvider

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

VECTOR_DB_PROVIDER_REGISTRY: dict[str, type[VectorDBProvider]] = {
    "milvus": MilvusProvider,
    # Add more providers here as they're implemented
    # "pinecone": PineconeProvider,
    # "weaviate": WeaviateProvider,
}


def register_vector_db_provider(name: str, provider_class: type[VectorDBProvider]) -> None:
    """Register a new vector database provider implementation.

    Args:
        name: Provider identifier (e.g. ``"milvus"``, ``"pinecone"``).
        provider_class: Class implementing ``VectorDBProvider``.

    Raises:
        ValueError: If *name* is empty.
    """
    name = name.lower()
    if name in VECTOR_DB_PROVIDER_REGISTRY:
        logger.warning(f"Overwriting existing vector DB provider registration: {name}")

    VECTOR_DB_PROVIDER_REGISTRY[name] = provider_class
    logger.info(f"Registered vector DB provider: {name} -> {provider_class.__name__}")


def get_available_vector_db_providers() -> list[str]:
    """Return the list of registered provider identifiers."""
    return list(VECTOR_DB_PROVIDER_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Primary factory
# ---------------------------------------------------------------------------


def _build_config_from_settings(
    *,
    host: str | None = None,
    port: int | None = None,
    alias: str | None = None,
    timeout: int | None = None,
    user: str | None = None,
    password: str | None = None,
    secure: bool | None = None,
    collection_name: str | None = None,
    collection_description: str | None = None,
    embedding_dimension: int | None = None,
    load_on_startup: bool | None = None,
    top_k: int | None = None,
    search_ef: int | None = None,
    output_fields: list[str] | None = None,
) -> VectorDBConfig:
    """Build a ``VectorDBConfig`` from keyword args, falling back to ``Settings``.

    Every parameter defaults to the corresponding value in the global
    ``Settings`` singleton when not explicitly provided.
    """
    settings = get_settings()

    conn = VectorDBConnectionConfig(
        host=host if host is not None else settings.milvus_host,
        port=port if port is not None else settings.milvus_port,
        alias=alias if alias is not None else settings.milvus_alias,
        timeout=timeout if timeout is not None else settings.milvus_timeout,
        user=user if user is not None else settings.milvus_user,
        password=password if password is not None else settings.milvus_password,
        secure=secure if secure is not None else settings.milvus_secure,
    )

    coll = VectorDBCollectionConfig(
        name=collection_name or settings.milvus_collection_name,
        description=collection_description or settings.milvus_collection_description,
        embedding_dimension=embedding_dimension or settings.milvus_embedding_dimension,
        load_on_startup=(
            load_on_startup if load_on_startup is not None else settings.milvus_load_on_startup
        ),
    )

    search = VectorDBSearchConfig(
        top_k=top_k if top_k is not None else settings.milvus_search_top_k,
        search_params={"ef": search_ef or settings.milvus_search_ef},
        output_fields=output_fields or settings.milvus_search_output_fields,
    )

    return VectorDBConfig(
        connection=conn,
        collection=coll,
        search=search,
        embedding_dimension=coll.embedding_dimension,
        embedding_normalized=True,
    )


def _resolve_embedding_provider(
    *,
    config: VectorDBConfig,
    explicit: EmbeddingProvider | None,
) -> EmbeddingProvider | None:
    """Return an embedding provider for the vector DB, or *None* on failure.

    If an *explicit* provider is given it is returned as-is (the caller is
    responsible for validating dimension compatibility).  Otherwise a Gemini
    provider is created using application ``Settings``.

    A creation failure is logged and *None* is returned — this is intentional:
    some provider implementations may still function for vector-only search,
    and the factory should not refuse to build a vector-DB provider just
    because an optional embedding helper could not be stood up.
    """
    if explicit is not None:
        return explicit

    settings = get_settings()
    try:
        return create_embedding_provider(
            provider_type="gemini",
            model_name=settings.gemini_embedding_model or "text-embedding-004",
            embedding_dimension=config.collection.embedding_dimension,
            task_type=EmbeddingTaskType.RETRIEVAL_QUERY,
        )
    except Exception:
        logger.warning(
            "Failed to create default Gemini embedding provider. "
            "Text-based search will not function until an embedding provider "
            "is supplied explicitly.",
            exc_info=True,
        )
        return None


def create_vector_db_provider(
    provider_type: str | None = None,
    config: VectorDBConfig | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    **kwargs: Any,
) -> VectorDBProvider:
    """Create a vector database provider instance.

    This is the **single entry-point** for building providers.  Configuration
    is resolved in this order of precedence:

    1. Explicit ``VectorDBConfig`` via the *config* parameter (full control).
    2. Individual keyword arguments (``host``, ``port``, ``collection_name``,
       ``embedding_dimension``, ``top_k``, …).
    3. Application ``Settings`` (``.env`` or environment variables).

    Args:
        provider_type: Provider identifier (default ``"milvus"``).
        config: Complete ``VectorDBConfig`` (takes highest precedence).
        embedding_provider: Optional embedding provider for text-based search.
            If *None* a Gemini provider is created from application settings.
        **kwargs: Individual configuration overrides (see
            :func:`_build_config_from_settings` for accepted keys).

    Returns:
        Configured ``VectorDBProvider`` instance.

    Raises:
        VectorDBConfigError: If *provider_type* is unsupported or
            configuration validation fails.
    """
    provider_type = (provider_type or "milvus").lower()

    if provider_type not in VECTOR_DB_PROVIDER_REGISTRY:
        supported = ", ".join(get_available_vector_db_providers())
        msg = f"Unsupported vector DB provider: '{provider_type}'. Supported: {supported}"
        logger.error(msg)
        raise VectorDBConfigError(msg)

    provider_class = VECTOR_DB_PROVIDER_REGISTRY[provider_type]

    logger.debug(
        "Creating vector DB provider",
        provider_type=provider_type,
        kwargs_keys=sorted(kwargs.keys()),
    )

    # ------------------------------------------------------------------
    # Resolve configuration
    # ------------------------------------------------------------------
    if config is not None:
        # Respect explicit config; still overlay any remaining kwargs
        if kwargs:
            logger.debug(
                "Merging kwargs into explicit VectorDBConfig", kwargs_keys=sorted(kwargs.keys())
            )
            config = _overlay_config(config, **kwargs)
    else:
        config = _build_config_from_settings(**kwargs)

    # ------------------------------------------------------------------
    # Resolve embedding provider
    # ------------------------------------------------------------------
    resolved_embedding = _resolve_embedding_provider(
        config=config,
        explicit=embedding_provider,
    )

    # ------------------------------------------------------------------
    # Instantiate
    # ------------------------------------------------------------------
    try:
        provider = provider_class(
            config=config,
            embedding_provider=resolved_embedding,
        )
        logger.info("Successfully created vector DB provider", provider_type=provider_type)
        return provider
    except Exception as exc:
        msg = f"Failed to initialize '{provider_type}' vector DB provider: {exc}"
        logger.error(msg, exc_info=True)
        raise VectorDBConfigError(msg) from exc


def _overlay_config(base: VectorDBConfig, **kwargs: Any) -> VectorDBConfig:
    """Return a new ``VectorDBConfig`` with *kwargs* overlaying *base*.

    This is a helper for callers that supply an explicit config **and**
    want to tweak one or two settings via kwargs.
    """
    conn = VectorDBConnectionConfig(
        host=kwargs.get("host", base.connection.host),
        port=kwargs.get("port", base.connection.port),
        alias=kwargs.get("alias", base.connection.alias),
        timeout=kwargs.get("timeout", base.connection.timeout),
        user=kwargs.get("user", base.connection.user),
        password=kwargs.get("password", base.connection.password),
        secure=kwargs.get("secure", base.connection.secure),
    )

    coll = VectorDBCollectionConfig(
        name=kwargs.get("collection_name", base.collection.name),
        description=kwargs.get("collection_description", base.collection.description),
        embedding_dimension=kwargs.get("embedding_dimension", base.collection.embedding_dimension),
        load_on_startup=kwargs.get("load_on_startup", base.collection.load_on_startup),
    )

    search = VectorDBSearchConfig(
        top_k=kwargs.get("top_k", base.search.top_k),
        search_params=kwargs.get("search_params", base.search.search_params),
        output_fields=kwargs.get("output_fields", base.search.output_fields),
    )

    return VectorDBConfig(
        connection=conn,
        collection=coll,
        search=search,
        embedding_dimension=coll.embedding_dimension,
        embedding_normalized=base.embedding_normalized,
    )


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=5)
def get_cached_vector_db_provider(
    provider_type: str = "milvus",
    host: str = "localhost",
    port: int = 19530,
    collection_name: str = "pharmaceutical_drugs",
    embedding_dimension: int = 768,
) -> VectorDBProvider:
    """Return a cached provider matching the given parameters.

    Only use this when configuration is **stable** for the lifetime of the
    process.  For dynamic configurations call :func:`create_vector_db_provider`
    directly.
    """
    return create_vector_db_provider(
        provider_type=provider_type,
        host=host,
        port=port,
        collection_name=collection_name,
        embedding_dimension=embedding_dimension,
    )


def clear_vector_db_provider_cache() -> None:
    """Clear the cached provider instances."""
    get_cached_vector_db_provider.cache_clear()
    logger.info("Vector DB provider cache cleared")


def get_default_vector_db_provider() -> VectorDBProvider:
    """Return a provider wired with application ``Settings`` defaults."""
    return create_vector_db_provider(provider_type="milvus")
