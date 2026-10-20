"""Abstract vector database provider interface, its configuration and search results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.logs import get_logger

from .exceptions import VectorDBConfigError

logger = get_logger(__name__)


class SearchMetricType(Enum):
    """
    Types of similarity metrics for vector search

    Different metrics optimize for different use cases
    """

    COSINE = "cosine"  # Cosine similarity (normalized vectors)
    INNER_PRODUCT = "inner_product"  # Inner product (IP) - works with normalized vectors
    EUCLIDEAN = "euclidean"  # L2 distance
    DOT_PRODUCT = "dot_product"  # Dot product


@dataclass
class VectorDBConnectionConfig:
    """
    Configuration for vector database connection

    This dataclass centralizes connection settings for vector databases.
    """

    # Connection settings
    host: str = "localhost"
    port: int = 19530
    alias: str = "default"
    timeout: int = 30

    # Authentication
    user: str = ""
    password: str = ""

    # Security (for TLS connections)
    secure: bool = False
    server_pem_path: str = ""
    server_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging/serialization"""
        return {
            "host": self.host,
            "port": self.port,
            "alias": self.alias,
            "timeout": self.timeout,
            "secure": self.secure,
        }


@dataclass
class VectorDBSearchConfig:
    """
    Configuration for vector database search operations

    This dataclass centralizes search-related settings.
    """

    # Search parameters
    top_k: int = 10  # Default number of results
    metric_type: SearchMetricType = SearchMetricType.INNER_PRODUCT

    # Index-specific search parameters
    search_params: dict[str, Any] = field(default_factory=dict)

    # Output configuration
    output_fields: list[str] = field(default_factory=list)
    include_metadata: bool = True

    # Filter support
    enable_filters: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "top_k": self.top_k,
            "metric_type": self.metric_type.value,
            "search_params": self.search_params,
            "output_fields": self.output_fields,
        }


@dataclass
class VectorDBCollectionConfig:
    """
    Configuration for vector database collection

    This dataclass centralizes collection-related settings.
    """

    # Collection settings
    name: str = "pharmaceutical_drugs"
    description: str = "Pharmaceutical drug database with embeddings for LLM lookup"
    embedding_dimension: int = 768  # Must match embedding provider dimension

    # Collection management
    load_on_startup: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "embedding_dimension": self.embedding_dimension,
            "load_on_startup": self.load_on_startup,
        }


@dataclass
class VectorDBConfig:
    """
    Complete configuration container for vector database provider settings

    This dataclass centralizes all configuration options for vector database providers,
    making it easy to manage and extend settings.
    """

    # Connection configuration
    connection: VectorDBConnectionConfig = field(default_factory=VectorDBConnectionConfig)

    # Collection configuration
    collection: VectorDBCollectionConfig = field(default_factory=VectorDBCollectionConfig)

    # Search configuration
    search: VectorDBSearchConfig = field(default_factory=VectorDBSearchConfig)

    # Embedding provider integration
    embedding_dimension: int = 768  # Must match collection embedding_dimension
    embedding_normalized: bool = True  # For IP metric compatibility

    # Additional provider-specific settings
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging/serialization"""
        return {
            "connection": self.connection.to_dict(),
            "collection": self.collection.to_dict(),
            "search": self.search.to_dict(),
            "embedding_dimension": self.embedding_dimension,
            "embedding_normalized": self.embedding_normalized,
        }

    def validate(self) -> bool:
        """
        Validate configuration consistency

        Returns:
            True if valid

        Raises:
            VectorDBConfigError: If configuration is invalid
        """
        # Validate embedding dimension matches collection dimension
        if self.embedding_dimension != self.collection.embedding_dimension:
            raise VectorDBConfigError(
                f"Embedding dimension ({self.embedding_dimension}) must match "
                f"collection embedding dimension ({self.collection.embedding_dimension})",
                details={
                    "config_embedding_dim": self.embedding_dimension,
                    "collection_embedding_dim": self.collection.embedding_dimension,
                },
            )

        # Validate metric type is appropriate for normalized embeddings
        if self.embedding_normalized and self.search.metric_type != SearchMetricType.INNER_PRODUCT:
            logger.warning(
                f"Normalized embeddings work best with INNER_PRODUCT metric, "
                f"but {self.search.metric_type.value} is configured"
            )

        return True


@dataclass
class SearchResult:
    """
    Standardized search result object from vector database providers

    This class ensures consistent result structure across different
    provider implementations.
    """

    # Result identification
    id: Any  # Result ID (can be int, str, etc. depending on provider)
    score: float  # Similarity score (higher is better)

    # Result data
    fields: dict[str, Any]  # Field values from the database
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "id": self.id,
            "score": self.score,
            "fields": self.fields,
            "metadata": self.metadata,
        }

    def get_field(self, field_name: str, default: Any = None) -> Any:
        """
        Get a field value by name

        Args:
            field_name: Name of the field to retrieve
            default: Default value if field doesn't exist

        Returns:
            Field value or default
        """
        return self.fields.get(field_name, default)


class VectorDBProvider(ABC):
    """Contract that every vector database backend implements."""

    def __init__(
        self,
        config: VectorDBConfig,
        *,
        embeddings: Any | None = None,
    ):
        """Validate ``config`` (raises VectorDBConfigError) and store the settings.

        ``embeddings`` turns query text into vectors for text-based search.
        """
        self.config = config
        self.collection_name = config.collection.name
        self.embedding_dimension = config.collection.embedding_dimension
        self.embeddings = embeddings
        self.provider_type = (
            self.__class__.__name__.lower().replace("provider", "").replace("vector", "")
        )

        config.validate()

        logger.info(
            f"Initialized {self.provider_type} vector database provider - "
            f"Collection: {self.collection_name}, "
            f"Dimension: {self.embedding_dimension}"
        )

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the database. Safe to call repeatedly. Raises VectorDBConnectionError."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection and release resources."""

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        limit: int | None = None,
        filter_expression: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Return the nearest results to ``query_vector``, best match first.

        ``limit`` defaults to ``config.search.top_k``. ``filter_expression`` uses
        the backend's own syntax. Raises VectorDBSearchError, or
        VectorDBDimensionMismatchError if the vector has the wrong dimension.
        """

    def validate_query_vector(self, query_vector: list[float]) -> bool:
        """Return True if the vector is non-empty and matches the embedding dimension."""
        if not query_vector:
            logger.warning("Query vector is empty")
            return False

        if len(query_vector) != self.embedding_dimension:
            logger.warning(
                f"Query vector dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(query_vector)}"
            )
            return False

        return True
