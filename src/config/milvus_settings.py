"""Milvus and its MinIO object storage settings.

Big-picture flow:
1. Describe how to connect to the Milvus vector database.
2. Describe the drug collection Milvus stores: its name and the embedding
   size it expects.
3. Describe how vector searches should behave (how many results, how
   thorough the search is).
4. Describe MinIO, the object-storage service Milvus uses internally to
   store its data files.
"""

from pydantic import BaseModel, Field

# See paths.py for what Pydantic's BaseModel and Field do.


# MilvusSettings: groups everything needed to connect to Milvus (the
# vector database behind drug search) and to MinIO (the storage service
# Milvus relies on), since in this deployment the two always travel together.
class MilvusSettings(BaseModel):
    """Connection, collection, search, and object storage fields."""

    # Milvus connection
    milvus_host: str = Field(
        default="localhost",
        description="Milvus server host. Use 'milvus-standalone' inside Docker Compose network.",
    )
    # milvus_host: where to find the Milvus server (see mongo_host in
    # mongodb_settings.py for why this differs between local runs and
    # Docker Compose).

    milvus_port: int = Field(
        default=19530,
        ge=1,
        le=65535,
        description="Milvus gRPC API port.",
    )
    # milvus_port: which network port Milvus is listening on. "gRPC" is the
    # particular network protocol Milvus's API uses.

    milvus_alias: str = Field(
        default="default",
        description="Milvus connection alias.",
    )
    # milvus_alias: a name Milvus's own client library uses internally to
    # label this connection; rarely needs to change.

    milvus_timeout: int = Field(
        default=30,
        ge=1,
        description="Milvus connection timeout in seconds.",
    )
    # milvus_timeout: how long to wait for Milvus to respond before giving up.

    milvus_user: str = Field(
        default="",
        description="Milvus authentication username (optional).",
    )
    # milvus_user: left blank by default because this project's local
    # Milvus setup does not require authentication.

    milvus_password: str = Field(
        default="",
        description="Milvus authentication password (optional).",
    )
    # milvus_password: paired with milvus_user above; also optional here.

    milvus_secure: bool = Field(
        default=False,
        description="Enable TLS/SSL for Milvus connection.",
    )
    # milvus_secure: whether the connection to Milvus should be encrypted.

    # Drug collection
    milvus_collection_name: str = Field(
        default="pharmaceutical_drugs",
        description="Milvus collection name. Must match ingestion config.",
    )
    # milvus_collection_name: a "collection" in Milvus is like a table in a
    # normal database - this is the one holding drug records. NOTE: it must
    # exactly match whatever name the data-ingestion pipeline used, or
    # search will find an empty or missing collection.

    milvus_collection_description: str = Field(
        default="Pharmaceutical drug database with embeddings for LLM lookup",
        description="Human-readable collection description.",
    )
    # milvus_collection_description: purely descriptive text, shown in
    # Milvus's own tools; does not affect app behavior.

    milvus_embedding_dimension: int = Field(
        default=768,
        ge=1,
        description="Milvus collection embedding dimension. Also the dimensions requested "
        "from the OpenAI embedding model, so it must equal the ingestion config's "
        "openai.dimensions.",
    )
    # milvus_embedding_dimension: how many numbers are in each embedding
    # vector. NOTE: this must match both the Milvus collection's actual
    # setup AND the OpenAI embedding client's requested size (see
    # src/adapters/llm_provider/openai/embeddings.py), or search breaks.

    milvus_load_on_startup: bool = Field(
        default=True,
        description="Load collection into memory automatically on connection.",
    )
    # milvus_load_on_startup: Milvus can search faster once a collection is
    # loaded into memory; this setting does that automatically on connect
    # instead of requiring a separate manual step.

    # Vector search
    milvus_search_top_k: int = Field(
        default=50,
        ge=1,
        description="Default number of nearest-neighbour results to return.",
    )
    # milvus_search_top_k: "nearest-neighbour" search means "find the K
    # stored vectors most similar to the search vector" - this is that K.

    milvus_search_ef: int = Field(
        default=64,
        ge=1,
        description="HNSW index search parameter (ef). Higher = better recall, slower search.",
    )
    # milvus_search_ef: a tuning knob for Milvus's search index (HNSW is the
    # indexing algorithm it uses). Higher values search more thoroughly
    # (better "recall", meaning fewer true matches are missed) at the cost
    # of speed.

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
    # milvus_search_output_fields: which drug record fields to fetch back
    # alongside each search match. "default_factory=lambda: [...]" is used
    # instead of a plain "default=[...]" because a mutable default like a
    # list would otherwise be shared between every Settings instance - a
    # classic Python trap. A "lambda" here is just a tiny, unnamed function
    # that builds a fresh list each time one is needed.

    # MinIO is Milvus's object storage backend in this deployment.
    minio_host: str = Field(
        default="localhost",
        description="MinIO server host. Use 'minio' inside Docker Compose network.",
    )
    # minio_host: where to find the MinIO server.

    minio_api_port: int = Field(
        default=9000,
        ge=1,
        le=65535,
        description="MinIO S3-compatible API port.",
    )
    # minio_api_port: MinIO speaks the same API style as Amazon S3 (a
    # well-known cloud storage service), so other tools can talk to it the
    # same way. This is that API's port.

    minio_console_port: int = Field(
        default=9001,
        ge=1,
        le=65535,
        description="MinIO web console port (browser UI).",
    )
    # minio_console_port: the port for MinIO's browser-based admin UI.

    minio_root_user: str = Field(
        default="minioadmin",
        description="MinIO root username / access key.",
    )
    # minio_root_user: the account MinIO uses to authenticate requests.

    minio_root_password: str = Field(
        default="minioadmin",
        description="MinIO root password / secret key.",
    )
    # minio_root_password: NOTE: this default is a well-known placeholder
    # meant only for local development - always override it for any real
    # deployment.

    minio_bucket_name: str = Field(
        default="milvus-bucket",
        description="Default MinIO bucket name for Milvus storage.",
    )
    # minio_bucket_name: a "bucket" is MinIO's term for a storage folder at
    # the top level; this is where Milvus keeps its data files.

    minio_secure: bool = Field(
        default=False,
        description="Enable TLS/SSL for MinIO connections.",
    )
    # minio_secure: whether the connection to MinIO should be encrypted.
