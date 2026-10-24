"""Milvus and its MinIO object storage settings."""

from pydantic import BaseModel, Field


class MilvusSettings(BaseModel):
    """Connection, collection, search, and object storage fields."""

    # Milvus connection
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

    # Drug collection
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
        description="Milvus collection embedding dimension. Also the dimensions requested "
        "from the OpenAI embedding model, so it must equal the ingestion config's "
        "openai.dimensions.",
    )

    milvus_load_on_startup: bool = Field(
        default=True,
        description="Load collection into memory automatically on connection.",
    )

    # Vector search
    milvus_search_top_k: int = Field(
        default=50,
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

    # MinIO is Milvus's object storage backend in this deployment.
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
