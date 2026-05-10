"""
Milvus Vector Database Provider Implementation

This module provides a comprehensive implementation of the ``VectorDBProvider``
interface using **Milvus** as the vector database backend.

Features:
- Connection management with health-check verification.
- Vector similarity search (IP / cosine / L2).
- Filter expression support with schema-aware output-field validation.
- Integration with embedding providers for ``search_by_text`` convenience.
- Full async/sync duality.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pymilvus import Collection, MilvusException, connections, utility

from src.config import get_settings
from src.logs import get_logger
from src.providers.embedding_provider import (
    EmbeddingProvider,
    EmbeddingTaskType,
)
from src.providers.vector_db_provider.base import (
    SearchResult,
    VectorDBConfig,
    VectorDBProvider,
)
from src.providers.vector_db_provider.exceptions import (
    VectorDBCollectionError,
    VectorDBConnectionError,
    VectorDBDimensionMismatchError,
    VectorDBProviderError,
    VectorDBSearchError,
)

logger = get_logger(__name__)


class MilvusProvider(VectorDBProvider):
    """Milvus vector database provider implementation.

    This provider supports Milvus vector database with both synchronous and
    asynchronous operations.  Default configuration is drawn from the
    application ``Settings`` singleton so that the same values used by the
    ingestion pipeline are applied at query time.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: VectorDBConfig | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Initialise the Milvus provider.

        Args:
            config: Complete ``VectorDBConfig``.  If *None* the factory builds
                one from application ``Settings``.
            embedding_provider: Optional embedding provider for text-based
                search.  If *None* a Gemini provider is created internally;
                if creation fails the provider will still function for
                vector-only search.
        """
        settings = get_settings()

        # Build default config from Settings when none supplied
        if config is None:
            from src.providers.vector_db_provider.base import (
                VectorDBCollectionConfig,
                VectorDBConnectionConfig,
                VectorDBSearchConfig,
            )

            conn = VectorDBConnectionConfig(
                host=settings.milvus_host,
                port=settings.milvus_port,
                alias=settings.milvus_alias,
                timeout=settings.milvus_timeout,
                user=settings.milvus_user,
                password=settings.milvus_password,
                secure=settings.milvus_secure,
            )
            coll = VectorDBCollectionConfig(
                name=settings.milvus_collection_name,
                description=settings.milvus_collection_description,
                embedding_dimension=settings.milvus_embedding_dimension,
                load_on_startup=settings.milvus_load_on_startup,
            )
            search = VectorDBSearchConfig(
                top_k=settings.milvus_search_top_k,
                search_params={"ef": settings.milvus_search_ef},
                output_fields=settings.milvus_search_output_fields,
            )
            config = VectorDBConfig(
                connection=conn,
                collection=coll,
                search=search,
                embedding_dimension=coll.embedding_dimension,
                embedding_normalized=True,
            )

        # ------------------------------------------------------------------
        # Resolve embedding provider
        # ------------------------------------------------------------------
        if embedding_provider is None:
            embedding_provider = self._create_default_embedding_provider(
                settings=settings,
                dimension=config.collection.embedding_dimension,
            )

        # Validate dimension agreement if an embedding provider *is* available
        if embedding_provider is not None:
            if embedding_provider.embedding_dimension != config.collection.embedding_dimension:
                raise VectorDBDimensionMismatchError(
                    f"Embedding provider dimension ({embedding_provider.embedding_dimension}) "
                    f"does not match collection dimension "
                    f"({config.collection.embedding_dimension})."
                )

        # Call parent (validates config)
        super().__init__(config, embedding_provider=embedding_provider)

        # ------------------------------------------------------------------
        # Internal state
        # ------------------------------------------------------------------
        self._collection: Collection | None = None
        self._is_connected: bool = False

        logger.info(
            "Milvus provider ready",
            collection=self.collection_name,
            dimension=self.embedding_dimension,
            has_embedding_provider=embedding_provider is not None,
        )

    @staticmethod
    def _create_default_embedding_provider(
        *,
        settings: Any,
        dimension: int,
    ) -> EmbeddingProvider | None:
        """Attempt to create a Gemini embedding provider from settings.

        Returns *None* if creation fails — the provider can still serve
        vector-only search requests.
        """
        from src.providers.embedding_provider import create_embedding_provider

        try:
            return create_embedding_provider(
                provider_type="gemini",
                model_name=settings.gemini_embedding_model or "text-embedding-004",
                embedding_dimension=dimension,
                task_type=EmbeddingTaskType.RETRIEVAL_QUERY,
            )
        except Exception:
            logger.warning(
                "Could not create default Gemini embedding provider. "
                "search_by_text() will raise an error unless an embedding "
                "provider is supplied at construction time.",
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to Milvus (async)."""
        try:
            await asyncio.to_thread(self.connect_sync)
        except VectorDBConnectionError:
            raise
        except Exception as exc:
            msg = f"Failed to connect to Milvus (async): {exc}"
            logger.error(msg, exc_info=True)
            raise VectorDBConnectionError(msg) from exc

    def connect_sync(self) -> None:
        """Connect to Milvus (synchronous).

        Idempotent — calling it on an already-connected instance is a no-op.
        """
        if self._is_connected:
            logger.debug("Already connected to Milvus")
            return

        conn_cfg = self.config.connection

        try:
            # Build connection kwargs
            params: dict[str, Any] = {
                "alias": conn_cfg.alias,
                "host": conn_cfg.host,
                "port": conn_cfg.port,
            }
            if conn_cfg.user:
                params["user"] = conn_cfg.user
                params["password"] = conn_cfg.password
            if conn_cfg.secure:
                params["secure"] = True
                if conn_cfg.server_pem_path:
                    params["server_pem_path"] = conn_cfg.server_pem_path
                if conn_cfg.server_name:
                    params["server_name"] = conn_cfg.server_name

            connections.connect(**params)
            logger.info(
                "Connecting to Milvus",
                host=conn_cfg.host,
                port=conn_cfg.port,
                alias=conn_cfg.alias,
            )

            # Health-check: verify the connection actually works
            try:
                _ = utility.list_collections(using=conn_cfg.alias)
                logger.info("Milvus connection verified")
            except Exception as verify_exc:
                msg = (
                    f"Milvus connection verification failed: {verify_exc}. "
                    f"Ensure Milvus is reachable at {conn_cfg.host}:{conn_cfg.port}"
                )
                logger.error(msg, exc_info=True)
                raise VectorDBConnectionError(msg) from verify_exc

            self._is_connected = True

            # Auto-load collection if configured
            if self.config.collection.load_on_startup:
                self._load_collection()

        except VectorDBConnectionError:
            self._is_connected = False
            raise
        except MilvusException as exc:
            msg = f"Milvus connection error: {exc}"
            logger.error(msg, exc_info=True)
            self._is_connected = False
            raise VectorDBConnectionError(msg) from exc
        except Exception as exc:
            msg = f"Unexpected error connecting to Milvus: {exc}"
            logger.error(msg, exc_info=True)
            self._is_connected = False
            raise VectorDBConnectionError(msg) from exc

    async def disconnect(self) -> None:
        """Disconnect from Milvus (async)."""
        try:
            await asyncio.to_thread(self.disconnect_sync)
        except VectorDBConnectionError:
            raise
        except Exception as exc:
            msg = f"Failed to disconnect from Milvus (async): {exc}"
            logger.error(msg, exc_info=True)
            raise VectorDBConnectionError(msg) from exc

    def disconnect_sync(self) -> None:
        """Disconnect from Milvus (synchronous).

        Idempotent — calling it when not connected is a no-op.
        """
        if not self._is_connected:
            logger.debug("Not connected to Milvus — nothing to disconnect")
            return

        try:
            self._collection = None
            connections.disconnect(alias=self.config.connection.alias)
            self._is_connected = False
            logger.info("Disconnected from Milvus", alias=self.config.connection.alias)
        except MilvusException as exc:
            msg = f"Milvus disconnect error: {exc}"
            logger.error(msg, exc_info=True)
            raise VectorDBConnectionError(msg) from exc

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> Collection:
        """Return the loaded ``Collection``, connecting & loading if needed.

        This is the **single chokepoint** that guards all ``self._collection``
        accesses — internal methods should call this instead of reading
        ``self._collection`` directly.
        """
        if not self._is_connected:
            self.connect_sync()

        if self._collection is None:
            self._load_collection()

        # After _load_collection, _collection is guaranteed non-None
        # (it raises on failure).
        assert self._collection is not None, "_load_collection should have raised"
        return self._collection

    def _load_collection(self) -> None:
        """Load the Milvus collection into memory.

        Raises:
            VectorDBCollectionError: If the collection doesn't exist or
                cannot be loaded.
        """
        if not self._is_connected:
            raise VectorDBCollectionError("Cannot load collection — not connected to Milvus")

        try:
            if not utility.has_collection(self.collection_name):
                raise VectorDBCollectionError(
                    f"Collection '{self.collection_name}' does not exist in Milvus. "
                    "Run the ingestion pipeline first."
                )

            self._collection = Collection(
                name=self.collection_name,
                using=self.config.connection.alias,
            )
            self._collection.load()
            logger.info(
                "Loaded collection into memory",
                collection=self.collection_name,
            )

        except VectorDBCollectionError:
            self._collection = None
            raise
        except MilvusException as exc:
            self._collection = None
            msg = f"Failed to load collection '{self.collection_name}': {exc}"
            logger.error(msg, exc_info=True)
            raise VectorDBCollectionError(msg) from exc
        except Exception as exc:
            self._collection = None
            msg = f"Unexpected error loading collection '{self.collection_name}': {exc}"
            logger.error(msg, exc_info=True)
            raise VectorDBCollectionError(msg) from exc

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query_vector: list[float],
        limit: int | None = None,
        filter_expression: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Perform vector similarity search (async)."""
        try:
            return await asyncio.to_thread(
                self.search_sync,
                query_vector,
                limit,
                filter_expression,
                output_fields,
                **kwargs,
            )
        except (VectorDBSearchError, VectorDBDimensionMismatchError):
            raise
        except Exception as exc:
            msg = f"Unexpected async search error: {exc}"
            logger.error(msg, exc_info=True)
            raise VectorDBSearchError(msg) from exc

    def search_sync(
        self,
        query_vector: list[float],
        limit: int | None = None,
        filter_expression: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Perform vector similarity search (synchronous).

        Returns results ordered by relevance (highest score first for IP
        metric).
        """
        # --- Dimension validation --------------------------------------------------
        if not self.validate_query_vector(query_vector):
            raise VectorDBDimensionMismatchError(
                f"Query vector dimension {len(query_vector)} does not match "
                f"collection dimension {self.embedding_dimension}."
            )

        # --- Ensure collection is available ----------------------------------------
        collection = self._ensure_collection()

        # --- Resolve parameters ----------------------------------------------------
        top_k = limit if limit is not None else self.config.search.top_k
        resolved_output: list[str] | None = (
            output_fields if output_fields is not None else self.config.search.output_fields
        )

        # Validate output_fields against the live schema
        resolved_output = self._validate_output_fields(collection, resolved_output)

        search_params: dict[str, Any] = dict(self.config.search.search_params)
        if "ef" not in search_params:
            search_params["ef"] = 64
        # Allow kwargs to override
        search_params.update(kwargs.get("search_params", {}))

        # --- Execute search --------------------------------------------------------
        logger.debug(
            "Searching Milvus",
            top_k=top_k,
            filter=filter_expression or "<none>",
            output_fields_count=len(resolved_output) if resolved_output else 0,
        )

        try:
            raw_results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expression,
                output_fields=resolved_output,
            )
        except MilvusException as exc:
            msg = f"Milvus search failed: {exc}"
            logger.error(msg, exc_info=True)
            raise VectorDBSearchError(msg) from exc
        except Exception as exc:
            msg = f"Unexpected search error: {exc}"
            logger.error(msg, exc_info=True)
            raise VectorDBSearchError(msg) from exc

        # --- Normalise results -----------------------------------------------------
        return self._format_search_results(raw_results, resolved_output)

    # ------------------------------------------------------------------
    # Text-based search convenience
    # ------------------------------------------------------------------

    async def search_by_text(
        self,
        query_text: str,
        limit: int | None = None,
        filter_expression: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Search using a natural-language query (auto-embeds)."""
        if self.embedding_provider is None:
            raise VectorDBSearchError(
                "search_by_text requires an embedding provider, but none is "
                "available. Pass an embedding_provider to the constructor."
            )

        try:
            logger.debug("Generating embedding for query", query=query_text[:80])
            query_embedding = await self.embedding_provider.generate_embedding(
                query_text,
                task_type=EmbeddingTaskType.RETRIEVAL_QUERY,
            )

            if not self.validate_query_vector(query_embedding):
                raise VectorDBDimensionMismatchError(
                    f"Generated embedding dimension {len(query_embedding)} != "
                    f"collection dimension {self.embedding_dimension}."
                )

            return await self.search(
                query_vector=query_embedding,
                limit=limit,
                filter_expression=filter_expression,
                output_fields=output_fields,
                **kwargs,
            )

        except (VectorDBSearchError, VectorDBDimensionMismatchError):
            raise
        except Exception as exc:
            msg = f"Text-based search failed: {exc}"
            logger.error(msg, exc_info=True)
            raise VectorDBSearchError(msg) from exc

    def search_by_text_sync(
        self,
        query_text: str,
        limit: int | None = None,
        filter_expression: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Search using a natural-language query — synchronous variant."""
        if self.embedding_provider is None:
            raise VectorDBSearchError(
                "search_by_text_sync requires an embedding provider, but none "
                "is available. Pass an embedding_provider to the constructor."
            )

        try:
            logger.debug("Generating embedding for query", query=query_text[:80])
            query_embedding = self.embedding_provider.generate_embedding_sync(
                query_text,
                task_type=EmbeddingTaskType.RETRIEVAL_QUERY,
            )

            if not self.validate_query_vector(query_embedding):
                raise VectorDBDimensionMismatchError(
                    f"Generated embedding dimension {len(query_embedding)} != "
                    f"collection dimension {self.embedding_dimension}."
                )

            return self.search_sync(
                query_vector=query_embedding,
                limit=limit,
                filter_expression=filter_expression,
                output_fields=output_fields,
                **kwargs,
            )

        except (VectorDBSearchError, VectorDBDimensionMismatchError):
            raise
        except Exception as exc:
            msg = f"Text-based search (sync) failed: {exc}"
            logger.error(msg, exc_info=True)
            raise VectorDBSearchError(msg) from exc

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    async def get_collection_info(self) -> dict[str, Any]:
        """Return metadata about the Milvus collection."""
        try:
            collection = self._ensure_collection()

            return {
                "name": self.collection_name,
                "num_entities": collection.num_entities,
                "description": collection.description,
                "schema": {
                    "fields": [
                        {
                            "name": field.name,
                            "type": str(field.dtype),
                            "description": field.description,
                        }
                        for field in collection.schema.fields
                    ]
                },
                "is_loaded": collection.has_index(),
            }
        except Exception as exc:
            msg = f"Failed to retrieve collection info: {exc}"
            logger.error(msg, exc_info=True)
            raise VectorDBProviderError(msg) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_output_fields(
        self,
        collection: Collection,
        requested: list[str] | None,
    ) -> list[str] | None:
        """Validate *requested* output fields against the collection schema.

        Returns a filtered list of **valid** scalar fields, or *None* when
        no valid fields remain.  Emits a warning when fields are dropped.
        """
        if not requested:
            return None

        # Empty list is not accepted by Milvus
        if len(requested) == 0:
            return None

        try:
            schema_field_names = {field.name for field in collection.schema.fields}
        except Exception as exc:
            logger.warning(
                "Could not introspect collection schema; returning None "
                "for output_fields to avoid errors.",
                error=str(exc),
            )
            return None

        # Keep only scalar fields that exist and aren't the pk or vector field
        valid = [f for f in requested if f in schema_field_names and f not in ("id", "embedding")]

        if len(valid) == 0:
            logger.warning(
                "No valid output_fields remain after schema validation. Returning None.",
                requested=requested,
                schema_fields=sorted(schema_field_names),
            )
            return None

        if len(valid) < len(requested):
            dropped = sorted(set(requested) - set(valid))
            logger.warning(
                "Dropped invalid output_fields",
                dropped=dropped,
                kept=valid,
            )

        return valid

    def _format_search_results(
        self,
        raw_results: Any,
        output_fields: list[str] | None,
    ) -> list[SearchResult]:
        """Convert Milvus search results into a list of ``SearchResult``.

        Handles *pymilvus* returning either ``SearchResult`` directly or an
        iterable of ``SearchResult`` (depending on pymilvus version).
        """
        results: list[SearchResult] = []

        # Normalise to an iterable of hits
        hits: Any = raw_results
        if hasattr(raw_results, "__iter__") and not isinstance(raw_results, list):
            # Single SearchResult with .ids, .distances, etc.
            hits = raw_results

        if hits is None:
            return results

        # pymilvus returns a list-of-lists: [ [Hit, Hit, ...] ]
        # where the outer list has one entry per query vector.
        if isinstance(hits, list) and len(hits) > 0:
            hit_list = hits[0] if isinstance(hits[0], list) else hits
        else:
            hit_list = hits if hasattr(hits, "__iter__") else []

        for hit in hit_list:
            fields: dict[str, Any] = {}
            if output_fields:
                for field_name in output_fields:
                    if hasattr(hit, "entity") and hasattr(hit.entity, "get"):
                        fields[field_name] = hit.entity.get(field_name)
                    elif hasattr(hit, field_name):
                        fields[field_name] = getattr(hit, field_name)

            score = float(hit.distance)

            results.append(
                SearchResult(
                    id=hit.id,
                    score=score,
                    fields=fields,
                    metadata={
                        "distance": score,
                        "collection": self.collection_name,
                    },
                )
            )

        logger.info("Search completed", result_count=len(results))
        return results
