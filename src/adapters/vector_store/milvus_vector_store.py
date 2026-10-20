"""Milvus implementation of the VectorDBProvider interface."""

from __future__ import annotations

import asyncio
from typing import Any

from pymilvus import Collection, DataType, MilvusException, connections, utility

from src.adapters.llm_provider.openai import OpenAIEmbeddings
from src.adapters.vector_store.base import (
    SearchResult,
    VectorDBCollectionConfig,
    VectorDBConfig,
    VectorDBConnectionConfig,
    VectorDBProvider,
    VectorDBSearchConfig,
)
from src.adapters.vector_store.exceptions import (
    VectorDBCollectionError,
    VectorDBConnectionError,
    VectorDBDimensionMismatchError,
    VectorDBSearchError,
)
from src.config import get_settings
from src.logs import get_logger

logger = get_logger(__name__)

DEFAULT_SEARCH_EF = 64


def _config_from_settings(settings: Any) -> VectorDBConfig:
    collection = VectorDBCollectionConfig(
        name=settings.milvus_collection_name,
        description=settings.milvus_collection_description,
        embedding_dimension=settings.milvus_embedding_dimension,
        load_on_startup=settings.milvus_load_on_startup,
    )
    return VectorDBConfig(
        connection=VectorDBConnectionConfig(
            host=settings.milvus_host,
            port=settings.milvus_port,
            alias=settings.milvus_alias,
            timeout=settings.milvus_timeout,
            user=settings.milvus_user,
            password=settings.milvus_password,
            secure=settings.milvus_secure,
        ),
        collection=collection,
        search=VectorDBSearchConfig(
            top_k=settings.milvus_search_top_k,
            search_params={"ef": settings.milvus_search_ef},
            output_fields=settings.milvus_search_output_fields,
        ),
        embedding_dimension=collection.embedding_dimension,
        embedding_normalized=True,
    )


class MilvusProvider(VectorDBProvider):
    """Vector search over a Milvus collection.

    Blocking pymilvus calls run in a worker thread so the public methods stay async.
    Settings come from ``config`` or, if omitted, from the application settings.
    ``embeddings`` defaults to the configured OpenAI embedding model, whose vectors
    must match the ones the collection was ingested with.
    """

    def __init__(
        self,
        config: VectorDBConfig | None = None,
        *,
        embeddings: OpenAIEmbeddings | None = None,
    ) -> None:
        settings = get_settings()
        config = config or _config_from_settings(settings)
        dimension = config.collection.embedding_dimension

        if embeddings is None:
            embeddings = OpenAIEmbeddings(settings.openai_embedding_model, dimension)
        elif embeddings.dimensions != dimension:
            raise VectorDBDimensionMismatchError(
                f"Embedding dimension ({embeddings.dimensions}) "
                f"does not match collection dimension ({dimension})."
            )

        super().__init__(config, embeddings=embeddings)
        self._collection: Collection | None = None
        self._is_connected = False

    # ==================== Connection ====================

    async def connect(self) -> None:
        await asyncio.to_thread(self._connect)

    async def disconnect(self) -> None:
        await asyncio.to_thread(self._disconnect)

    async def check_collection(self) -> None:
        """Confirm the collection can load and its vector dimension matches settings."""
        await asyncio.to_thread(self._check_collection)

    def _check_collection(self) -> None:
        collection = self._ensure_collection()
        vector_fields = [
            field for field in collection.schema.fields if field.dtype == DataType.FLOAT_VECTOR
        ]
        if not vector_fields:
            raise VectorDBCollectionError(
                f"Collection '{self.collection_name}' has no float vector field"
            )
        actual_dimension = int(vector_fields[0].params["dim"])
        if actual_dimension != self.embedding_dimension:
            raise VectorDBDimensionMismatchError(
                f"Collection '{self.collection_name}' has dimension {actual_dimension}; "
                f"expected {self.embedding_dimension}. Check the ingestion configuration."
            )

    def _connect(self) -> None:
        if self._is_connected:
            return

        conn = self.config.connection
        params: dict[str, Any] = {"alias": conn.alias, "host": conn.host, "port": conn.port}
        if conn.user:
            params.update(user=conn.user, password=conn.password)
        if conn.secure:
            params["secure"] = True
            if conn.server_pem_path:
                params["server_pem_path"] = conn.server_pem_path
            if conn.server_name:
                params["server_name"] = conn.server_name

        try:
            connections.connect(**params)
            utility.list_collections(using=conn.alias)  # fails if the server is unreachable
            self._is_connected = True
            if self.config.collection.load_on_startup:
                self._load_collection()
        except VectorDBCollectionError:
            raise  # connected, but the collection is missing or cannot be loaded
        except Exception as exc:
            self._is_connected = False
            logger.error(f"Milvus connection failed: {exc}", exc_info=True)
            raise VectorDBConnectionError(
                f"Could not connect to Milvus at {conn.host}:{conn.port}: {exc}"
            ) from exc

    def _disconnect(self) -> None:
        if not self._is_connected:
            return
        try:
            self._collection = None
            connections.disconnect(alias=self.config.connection.alias)
            self._is_connected = False
        except MilvusException as exc:
            raise VectorDBConnectionError(f"Milvus disconnect error: {exc}") from exc

    def _load_collection(self) -> Collection:
        alias = self.config.connection.alias
        try:
            if not utility.has_collection(self.collection_name, using=alias):
                raise VectorDBCollectionError(
                    f"Collection '{self.collection_name}' does not exist in Milvus. "
                    "Run the ingestion pipeline first."
                )
            collection = Collection(name=self.collection_name, using=alias)
            collection.load()
        except VectorDBCollectionError:
            raise
        except Exception as exc:
            raise VectorDBCollectionError(
                f"Failed to load collection '{self.collection_name}': {exc}"
            ) from exc

        self._collection = collection
        logger.info("Loaded collection into memory", collection=self.collection_name)
        return collection

    def _ensure_collection(self) -> Collection:
        if not self._is_connected:
            self._connect()
        if self._collection is None:
            return self._load_collection()
        return self._collection

    # ==================== Search ====================

    async def search(
        self,
        query_vector: list[float],
        limit: int | None = None,
        filter_expression: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        return await asyncio.to_thread(
            self._search, query_vector, limit, filter_expression, output_fields, **kwargs
        )

    def _search(
        self,
        query_vector: list[float],
        limit: int | None,
        filter_expression: str | None,
        output_fields: list[str] | None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        if not self.validate_query_vector(query_vector):
            raise VectorDBDimensionMismatchError(
                f"Query vector dimension {len(query_vector)} does not match "
                f"collection dimension {self.embedding_dimension}."
            )

        collection = self._ensure_collection()
        requested = output_fields if output_fields is not None else self.config.search.output_fields
        output_fields = self._valid_output_fields(collection, requested)
        search_params = {
            "ef": DEFAULT_SEARCH_EF,
            **self.config.search.search_params,
            **kwargs.get("search_params", {}),
        }

        try:
            raw_results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=limit if limit is not None else self.config.search.top_k,
                expr=filter_expression,
                output_fields=output_fields,
            )
        except Exception as exc:
            logger.error(f"Milvus search failed: {exc}", exc_info=True)
            raise VectorDBSearchError(f"Milvus search failed: {exc}") from exc

        return self._to_search_results(raw_results, output_fields)

    async def search_by_text(
        self,
        query_text: str,
        limit: int | None = None,
        filter_expression: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Embed ``query_text`` and search with it.

        Raises:
            VectorDBSearchError: If embedding the query fails.
        """
        try:
            query_vector = await self.embeddings.embed(query_text)
        except Exception as exc:
            logger.error(f"Query embedding failed: {exc}", exc_info=True)
            raise VectorDBSearchError(f"Text-based search failed: {exc}") from exc

        return await self.search(query_vector, limit, filter_expression, output_fields, **kwargs)

    # ==================== Helpers ====================

    @staticmethod
    def _valid_output_fields(
        collection: Collection, requested: list[str] | None
    ) -> list[str] | None:
        """Keep the requested fields that exist in the schema, excluding the key and vector.

        Returns None if none remain, because Milvus rejects an empty list.
        """
        if not requested:
            return None

        schema_fields = {field.name for field in collection.schema.fields}
        valid = [name for name in requested if name in schema_fields - {"id", "embedding"}]

        if len(valid) < len(requested):
            logger.warning(
                "Dropped invalid output_fields",
                dropped=sorted(set(requested) - set(valid)),
                kept=valid,
            )
        return valid or None

    def _to_search_results(
        self, raw_results: Any, output_fields: list[str] | None
    ) -> list[SearchResult]:
        """Convert pymilvus hits for the single query vector into SearchResult objects."""
        hits = raw_results[0] if raw_results else []
        results = [
            SearchResult(
                id=hit.id,
                score=float(hit.distance),
                fields={name: hit.entity.get(name) for name in output_fields or []},
                metadata={"distance": float(hit.distance), "collection": self.collection_name},
            )
            for hit in hits
        ]
        logger.info("Search completed", result_count=len(results))
        return results
