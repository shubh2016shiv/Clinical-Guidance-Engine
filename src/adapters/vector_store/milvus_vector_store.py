"""Connect to Milvus and search the existing drug collection.

Big-picture flow:
1. Build a Milvus-specific config from the app's settings.
2. Connect to Milvus using its SYNCHRONOUS client library, but run every
   call through ``asyncio.to_thread`` so it does not freeze the rest of
   the async application while Milvus is working.
3. Make sure the drug collection exists, is loaded into memory, and has
   the expected vector size, before trusting it to answer searches.
4. Offer ``search`` (by vector) and ``search_by_text`` (embeds the text
   first, then searches) as the two ways the rest of the app can query
   Milvus for similar drugs.
"""

from __future__ import annotations

import asyncio

# asyncio is Python's standard library for writing "async" code - code
# that can pause while waiting on slow operations (network, disk, or here,
# a blocking library call) instead of freezing the whole program.
import threading

# threading provides a Lock: a "only one thread at a time past this point"
# gate. It is needed here (rather than an asyncio lock) because the
# blocking Milvus calls run on background threads via asyncio.to_thread,
# so the code that must be protected is not running in the event loop.
from typing import Any

from pymilvus import DataType, MilvusClient

# pymilvus is Milvus's official Python client library. Unlike the OpenAI
# SDK used elsewhere in this project, MilvusClient is SYNCHRONOUS (it
# blocks the calling thread until it gets a reply) - this file wraps it so
# the rest of the app can still treat it like an async operation.
from src.adapters.llm_provider.openai import OpenAITextEmbeddingClient
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
# DEFAULT_SEARCH_EF: how hard the HNSW index searches when nothing else is
# configured. Milvus requires "ef" to be at least as large as the number of
# results asked for, so _search below raises it to match a bigger limit.


# _config_from_settings: builds a VectorDBConfig (the generic config shape
# from base.py) out of this app's Milvus-specific settings. Needed so
# MilvusProvider can be created with no arguments at all in normal use,
# while still allowing a custom config to be passed in for tests.
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


# MilvusProvider: the concrete vector-database implementation the rest of
# the app actually uses for drug search. It builds on VectorDBProvider
# (the shared contract from base.py) by adding Milvus's own connection,
# collection-loading, and search logic.
class MilvusProvider(VectorDBProvider):
    """Use MilvusClient for searches without blocking the async application."""

    def __init__(
        self,
        config: VectorDBConfig | None = None,
        *,
        embeddings: OpenAITextEmbeddingClient | None = None,
    ) -> None:
        # The "*" makes "embeddings" keyword-only.
        settings = get_settings()
        config = config or _config_from_settings(settings)
        # Step 1: use the given config, or build one from app settings if
        # none was provided - this lets MilvusProvider() work with zero
        # arguments in normal use.
        dimension = config.collection.embedding_dimension
        self._owns_embeddings = embeddings is None
        # self._owns_embeddings: remembers whether THIS object created its
        # own embedding client (True) or was handed an existing one to
        # reuse (False) - used by disconnect() below to decide whether it
        # is this object's job to close it.
        if embeddings is None:
            # Step 2a: no embedding client was given - build one sized to
            # match this collection's expected dimension.
            embeddings = OpenAITextEmbeddingClient(settings.openai_embedding_model, dimension)
        elif embeddings.dimensions != dimension:
            # Step 2b: an embedding client WAS given, but if its vector
            # size does not match the collection's, every search using it
            # would be broken - catch that mismatch immediately instead of
            # letting it fail confusingly deep inside a Milvus call later.
            raise VectorDBDimensionMismatchError(
                f"Embedding dimension ({embeddings.dimensions}) "
                f"does not match collection dimension ({dimension})."
            )
        super().__init__(config, embeddings=embeddings)
        # Step 3: let the parent class validate the config, store it, and
        # log the "initialized" message.
        self._client: MilvusClient | None = None
        # self._client: the actual Milvus connection, created later by
        # connect(). Starts as None because no connection exists yet.
        self._collection_description: dict[str, Any] | None = None
        # self._collection_description: a cached copy of the collection's
        # schema (field names/types), fetched once and reused.
        self._collection_loaded = False
        # self._collection_loaded: whether the collection has already been
        # loaded into Milvus's memory for fast search, so it is not
        # reloaded unnecessarily on every call.
        self._closed = False
        # self._closed: set once disconnect() has closed an embedding client
        # this provider owns. That client cannot be reopened, so the
        # provider refuses further work with a clear message instead of
        # failing later with a confusing "Connection error".
        self._state_lock = threading.RLock()
        # self._state_lock: guards connecting and collection loading, which
        # several background threads can reach at the same time. It is an
        # RLock ("reentrant") because _connect holds it while calling
        # _load_collection, and the same thread must be allowed back in.

    async def connect(self) -> None:
        await asyncio.to_thread(self._connect)
        # "asyncio.to_thread(fn)" runs the ordinary, BLOCKING function "fn"
        # on a separate background thread and lets the async program keep
        # going while it waits - this is how a synchronous library like
        # MilvusClient gets used safely from async code, without freezing
        # everything else the app is doing at the same time.

    def _raise_if_closed(self) -> None:
        """Refuse work once disconnect() has closed an owned embedding client."""
        if self._closed:
            raise VectorDBConnectionError(
                "This MilvusProvider was closed by disconnect() and cannot be reused. "
                "Create a new MilvusProvider instead."
            )

    def _connect(self) -> None:
        # This is a plain, synchronous method - it is only ever run inside
        # a background thread via asyncio.to_thread above, never awaited
        # directly.
        self._raise_if_closed()
        if self._client is not None:
            # Step 1: already connected - do nothing (connect() is meant
            # to be safe to call repeatedly). Checking here, before taking
            # the lock below, keeps the common "already connected" case
            # free of any locking cost.
            return
        with self._state_lock:
            # Step 2: only one thread may build the connection. Without
            # this, several threads calling connect() at the same time
            # would each create their own MilvusClient, and every one but
            # the last would be silently dropped while still holding an
            # open network connection.
            if self._client is not None:
                # Another thread finished connecting while this one waited
                # for the lock, so there is nothing left to do. This second
                # check is what makes the cheap check above safe.
                return
            self._connect_locked()

    def _connect_locked(self) -> None:
        # Runs with self._state_lock already held by the calling thread.
        connection = self.config.connection
        scheme = "https" if connection.secure else "http"
        # scheme: "https" for an encrypted connection, "http" otherwise -
        # this is a conditional expression, a compact "if/else" that
        # produces a value instead of running a statement.
        options: dict[str, Any] = {
            "uri": f"{scheme}://{connection.host}:{connection.port}",
            "user": connection.user,
            "password": connection.password,
            "timeout": connection.timeout,
        }
        # options: the keyword arguments that will be passed to
        # MilvusClient(...) below, built up step by step.
        if connection.server_pem_path:
            options["server_pem_path"] = connection.server_pem_path
        if connection.server_name:
            options["server_name"] = connection.server_name
        # Step 1: only add the TLS-specific options if they were actually
        # configured, since most local setups do not need them.
        client = None
        try:
            # Step 2: actually create the client AND immediately check
            # that it truly works, by listing collections - creating the
            # object alone does not prove Milvus is reachable.
            client = MilvusClient(**options)
            client.list_collections()
        except Exception as exc:
            # Step 3: if anything above failed, make sure we do not leak a
            # half-open connection, then report a clear, specific error.
            if client is not None:
                client.close()
            raise VectorDBConnectionError(
                f"Could not connect to Milvus at {connection.host}:{connection.port}: {exc}"
            ) from exc
        # Step 4: only keep the connection once it is confirmed working.
        self._client = client
        if self.config.collection.load_on_startup:
            # Step 5: optionally load the collection into memory right
            # away, so the first real search is not slowed down by a
            # one-time loading step.
            self._load_collection()

    async def disconnect(self) -> None:
        try:
            await asyncio.to_thread(self._disconnect)
        finally:
            # "finally" runs whether or not the try block raised, so the
            # embedding client is always closed if this provider owns it,
            # even if closing the Milvus connection itself failed.
            if self._owns_embeddings and not self._closed:
                await self.embeddings.close()
                # Closing an owned embedding client cannot be undone, so
                # the provider is now permanently spent. Recording that
                # lets later calls fail with a clear explanation rather
                # than a confusing network error from the closed client.
                # A provider given someone else's embedding client is left
                # reusable, because nothing of its own was destroyed.
                self._closed = True

    def _disconnect(self) -> None:
        client = self._client
        self._client = None
        self._collection_description = None
        self._collection_loaded = False
        # Step 1: clear all cached state FIRST, so this object is in a
        # clean "not connected" state even if closing the client below
        # raises an error.
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                raise VectorDBConnectionError(f"Milvus disconnect error: {exc}") from exc

    def _ensure_client(self) -> MilvusClient:
        if self._client is None:
            self._connect()
        assert self._client is not None
        # "assert" is a sanity check for the programmer, not normal error
        # handling: it tells a type checker "trust me, this is not None
        # here" after we just connected above, and would raise
        # AssertionError only if that assumption were somehow wrong (a bug).
        return self._client

    def _load_collection(self) -> dict[str, Any]:
        if self._collection_loaded and self._collection_description is not None:
            # Step 1: if the collection was already loaded and described,
            # reuse the cached description instead of asking Milvus again.
            # This check sits outside the lock below so that ordinary
            # searches, which always take this path, never wait on it.
            return self._collection_description
        with self._state_lock:
            # Step 2: only one thread may do the first load, so concurrent
            # first searches cannot each issue their own describe/load
            # round trip to Milvus.
            if self._collection_loaded and self._collection_description is not None:
                return self._collection_description
            return self._load_collection_locked()

    def _load_collection_locked(self) -> dict[str, Any]:
        # Runs with self._state_lock already held by the calling thread.
        client = self._ensure_client()
        try:
            if not client.has_collection(collection_name=self.collection_name):
                # Step 1: if the collection does not exist at all, this is
                # almost always because the data-ingestion pipeline has
                # not been run yet - say so directly instead of letting a
                # generic Milvus error surface.
                raise VectorDBCollectionError(
                    f"Collection '{self.collection_name}' does not exist in Milvus. "
                    "Run the ingestion pipeline first."
                )
            description = client.describe_collection(collection_name=self.collection_name)
            # description: the collection's schema (its field names/types).
            if not self._collection_loaded:
                # Step 2: load the collection into memory only once - this
                # makes later searches fast, and does not need repeating.
                client.load_collection(collection_name=self.collection_name)
                self._collection_loaded = True
                logger.info("Loaded collection into memory", collection=self.collection_name)
        except VectorDBCollectionError:
            # Let our OWN already-specific error pass through unchanged.
            raise
        except Exception as exc:
            # Anything else unexpected gets wrapped into our own exception
            # type, so callers never need to catch raw Milvus exceptions.
            raise VectorDBCollectionError(
                f"Failed to load collection '{self.collection_name}': {exc}"
            ) from exc
        self._collection_description = description
        return description

    async def check_collection(self) -> None:
        """Check that the collection can load and has the right vector size."""
        await asyncio.to_thread(self._check_collection)

    def _check_collection(self) -> None:
        description = self._load_collection()
        vector_fields = [
            field
            for field in description["fields"]
            if field["name"] == "embedding" and field["type"] == DataType.FLOAT_VECTOR
        ]
        # vector_fields: every field in the collection's schema that is
        # specifically the "embedding" float-vector column - normally
        # exactly one, found here with a list comprehension (a compact way
        # to build a filtered list in one line instead of a manual loop).
        if not vector_fields:
            # Step 1: if the collection has no vector field at all, it was
            # set up incorrectly - fail clearly.
            raise VectorDBCollectionError(
                f"Collection '{self.collection_name}' has no float vector field"
            )
        actual_dimension = int(vector_fields[0]["params"]["dim"])
        # actual_dimension: the REAL vector size Milvus has stored,
        # straight from its own schema - this is compared against what
        # this app expects, below.
        if actual_dimension != self.embedding_dimension:
            # Step 2: catch a mismatch between what this app's config
            # expects and what Milvus actually has, with a message that
            # points directly at the likely cause (ingestion configuration).
            raise VectorDBDimensionMismatchError(
                f"Collection '{self.collection_name}' has dimension {actual_dimension}; "
                f"expected {self.embedding_dimension}. Check the ingestion configuration."
            )

    async def get_client(self) -> MilvusClient:
        """Return the connected client for drug-name hybrid searches."""
        return await asyncio.to_thread(self._get_loaded_client)

    def _get_loaded_client(self) -> MilvusClient:
        self._load_collection()
        return self._ensure_client()

    async def describe_collection(self) -> dict[str, Any]:
        """Return the collection schema, including its BM25 functions."""
        return await asyncio.to_thread(self._load_collection)

    async def count_entities(self) -> int:
        """Return the current row count from Milvus."""
        return await asyncio.to_thread(self._count_entities)

    def _count_entities(self) -> int:
        client = self._get_loaded_client()
        try:
            rows = client.query(
                collection_name=self.collection_name, filter="", output_fields=["count(*)"]
            )
            # An empty filter ("") matches every row, so this effectively
            # counts the whole collection using Milvus's own count function.
            return int(rows[0]["count(*)"])
        except Exception as exc:
            raise VectorDBCollectionError(
                f"Could not count rows in '{self.collection_name}': {exc}"
            ) from exc

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
        # "asyncio.to_thread(fn, *args, **kwargs)" also accepts arguments
        # to pass straight through to "fn" when it runs on the background
        # thread, which is how this async wrapper hands everything off to
        # the synchronous _search below.

    def _search(
        self,
        query_vector: list[float],
        limit: int | None,
        filter_expression: str | None,
        output_fields: list[str] | None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        if not self.validate_query_vector(query_vector):
            # Step 1: reject an obviously wrong vector (empty, or wrong
            # size) before spending a network round trip sending it to
            # Milvus - validate_query_vector is the shared check from
            # base.py.
            raise VectorDBDimensionMismatchError(
                f"Query vector dimension {len(query_vector)} does not match "
                f"collection dimension {self.embedding_dimension}."
            )
        client = self._get_loaded_client()
        assert self._collection_description is not None
        # Sanity check for the programmer: _get_loaded_client() above
        # guarantees the collection has been described by this point.
        requested = output_fields if output_fields is not None else self.config.search.output_fields
        # requested: use the fields the CALLER asked for, or fall back to
        # the configured defaults if the caller did not specify any.
        selected_fields = self._valid_output_fields(self._collection_description, requested)
        # Step 2: drop any requested field names that do not actually
        # exist in this collection's schema (see _valid_output_fields below).
        resolved_limit = limit if limit is not None else self.config.search.top_k
        # resolved_limit: how many results this search will ask for - the
        # caller's limit, or the configured default. Worked out once here
        # because the "ef" rule below depends on it as well.
        search_params = {"ef": DEFAULT_SEARCH_EF, **self.config.search.search_params}
        search_params.update(kwargs.get("search_params", {}))
        # Step 3: build the final search tuning parameters - start from a
        # built-in default, let the configured defaults override it, then
        # let any "search_params" passed into THIS specific call override
        # that again. "{**a, **b}" merges two dicts, with "b" winning on
        # shared keys; ".update(...)" then merges a third source in the
        # same "later wins" way.
        configured_ef = search_params.get("ef")
        if configured_ef is not None and configured_ef < resolved_limit:
            # Step 4: Milvus rejects an HNSW search outright when "ef" is
            # smaller than the number of results requested, so asking for
            # more results than "ef" would fail the whole query rather
            # than simply returning fewer rows. Raising "ef" to the limit
            # keeps the request valid. Only an existing "ef" is adjusted,
            # so index types that tune with other parameters are untouched.
            search_params["ef"] = resolved_limit
            logger.debug(
                "Raised HNSW ef to match the requested limit",
                requested_limit=resolved_limit,
                configured_ef=configured_ef,
            )
        try:
            raw_results = client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field="embedding",
                search_params=search_params,
                limit=resolved_limit,
                filter=filter_expression or "",
                output_fields=selected_fields,
            )
            # "anns_field" tells Milvus which schema field holds the
            # vectors to compare against ("embedding" in this collection).
            # Milvus accepts a BATCH of query vectors at once, which is why
            # "data" here is a list containing just our one query_vector.
        except Exception as exc:
            logger.error(f"Milvus search failed: {exc}", exc_info=True)
            raise VectorDBSearchError(f"Milvus search failed: {exc}") from exc
        hits = raw_results[0] if raw_results else []
        # hits: the results for OUR one query vector - since Milvus
        # returns one list of results PER query vector sent, and we only
        # sent one, we take the first (and only) list of matches.
        results = [
            SearchResult(
                id=hit["id"],
                score=float(hit["distance"]),
                fields={name: hit["entity"].get(name) for name in selected_fields or []},
                metadata={"distance": float(hit["distance"]), "collection": self.collection_name},
            )
            for hit in hits
        ]
        # results: Milvus's raw hit dictionaries translated into this
        # app's own SearchResult shape, one per match, using a list
        # comprehension. The "fields={...}" part is itself a DICT
        # comprehension, building {field_name: value} for every requested
        # output field on each hit.
        logger.info("Search completed", result_count=len(results))
        return results

    async def search_by_text(
        self,
        query_text: str,
        limit: int | None = None,
        filter_expression: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Embed the text, then search the existing collection."""
        # This path uses the embedding client before touching Milvus, so it
        # needs its own closed check: without it, a provider already shut
        # down by disconnect() would fail inside the closed embedding
        # client with an unhelpful "Connection error".
        self._raise_if_closed()
        try:
            # Step 1: turn the search text into a vector first, using the
            # same embedding model/dimension this provider was configured with.
            query_vector = await self.embeddings.create_text_embedding(query_text)
        except Exception as exc:
            raise VectorDBSearchError(f"Text-based search failed: {exc}") from exc
        # Step 2: once we have a vector, this is just a normal vector search.
        return await self.search(query_vector, limit, filter_expression, output_fields, **kwargs)

    @staticmethod
    def _valid_output_fields(
        description: dict[str, Any], requested: list[str] | None
    ) -> list[str] | None:
        # "@staticmethod" means this method does not need "self" - it only
        # works with the arguments it is given, so it can be called
        # without a specific MilvusProvider instance if ever needed.
        if not requested:
            # Step 1: nothing was requested - nothing to validate.
            return None
        schema_fields = {field["name"] for field in description["fields"]}
        # schema_fields: every real field name in this collection's
        # schema, built as a SET (not a list) using a set comprehension,
        # so membership checks below ("in schema_fields") are fast.
        valid = [name for name in requested if name in schema_fields - {"id", "embedding"}]
        # valid: only the requested names that both (a) actually exist in
        # the schema, and (b) are not the internal "id"/"embedding"
        # columns, which are not meant to be returned as regular output.
        # "schema_fields - {"id", "embedding"}" is SET subtraction: "every
        # schema field EXCEPT these two."
        if len(valid) < len(requested):
            # Step 2: if some requested names were dropped, warn about it
            # (likely a typo in configuration) rather than failing silently
            # or crashing the whole search over it.
            logger.warning(
                "Dropped invalid output_fields",
                dropped=sorted(set(requested) - set(valid)),
                kept=valid,
            )
        return valid or None
        # "valid or None" means: return the list if it has anything in it,
        # or None if it ended up empty - callers treat None as "no specific
        # fields requested."
