"""Abstract vector database provider interface, its configuration and search results.

Big-picture flow:
1. Describe the different pieces of configuration a vector database needs:
   how to connect, how to search, and which collection to use.
2. Describe ``SearchResult``: one consistent shape for a single match,
   no matter which backend (Milvus, etc.) produced it.
3. Define ``VectorDBProvider``: the "contract" every vector database
   backend must follow, plus shared validation logic every backend can reuse.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

# ABC (Abstract Base Class) lets us define a "template" class that cannot
# be created directly - only its subclasses can be. abstractmethod marks a
# method every subclass MUST implement, or Python raises an error.
from dataclasses import dataclass, field

# @dataclass is a shortcut that writes the boring parts of a class (like
# __init__) for us, just from the fields listed below. "field" lets us give
# a field a safer default than writing "= {}" or "= []" directly (see the
# note on VectorDBSearchConfig below).
from enum import Enum

# Enum creates a small, fixed set of named choices (like a multiple-choice
# question) instead of plain strings, so a typo is caught by your editor or
# type checker instead of silently picking the wrong search behavior.
from typing import Any

from src.logs import get_logger

from .exceptions import VectorDBConfigError

logger = get_logger(__name__)


# SearchMetricType: the fixed set of ways two vectors' similarity can be
# measured. Needed so the search config can only ever hold one of these
# known, valid choices instead of an arbitrary typed string.
class SearchMetricType(Enum):
    """
    Types of similarity metrics for vector search

    Different metrics optimize for different use cases
    """

    COSINE = "cosine"  # Cosine similarity (normalized vectors)
    INNER_PRODUCT = "inner_product"  # Inner product (IP) - works with normalized vectors
    EUCLIDEAN = "euclidean"  # L2 distance
    DOT_PRODUCT = "dot_product"  # Dot product


# VectorDBConnectionConfig: a plain data container holding everything
# needed to open a connection to a vector database (host, port,
# credentials, TLS settings).
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
        # NOTE: "password" is deliberately left out here so it never ends
        # up in a log line.
        return {
            "host": self.host,
            "port": self.port,
            "alias": self.alias,
            "timeout": self.timeout,
            "secure": self.secure,
        }


# VectorDBSearchConfig: a plain data container holding how searches should
# behave by default (how many results, which similarity metric, etc.).
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
    # "field(default_factory=dict)" is needed instead of "= {}" because a
    # mutable default like a dict would otherwise be shared between every
    # instance of this class - a classic Python trap this avoids.

    # Output configuration
    output_fields: list[str] = field(default_factory=list)
    # Same reasoning as above, but for an empty list default.
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


# VectorDBCollectionConfig: a plain data container describing WHICH
# collection (Milvus's equivalent of a database table) to use, and its
# expected vector size.
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


# VectorDBConfig: the single top-level config object that bundles the
# three groups above (connection, collection, search) plus a couple of
# cross-cutting settings. Needed so a vector database provider only needs
# ONE object passed in, instead of three separate ones.
@dataclass
class VectorDBConfig:
    """
    Complete configuration container for vector database provider settings

    This dataclass centralizes all configuration options for vector database providers,
    making it easy to manage and extend settings.
    """

    # Connection configuration
    connection: VectorDBConnectionConfig = field(default_factory=VectorDBConnectionConfig)
    # "field(default_factory=VectorDBConnectionConfig)" builds a FRESH
    # VectorDBConnectionConfig() for every VectorDBConfig instance, instead
    # of sharing one mutable instance between all of them.

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
            # Step 1: these two numbers MUST agree, because the embedding
            # client and the Milvus collection need to produce/expect
            # vectors of the exact same length - a mismatch here would
            # otherwise surface later as a confusing Milvus error instead
            # of this clear one.
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
            # Step 2: this is a softer problem - not strictly broken, but
            # likely to give worse search quality, so it only warns instead
            # of raising.
            logger.warning(
                f"Normalized embeddings work best with INNER_PRODUCT metric, "
                f"but {self.search.metric_type.value} is configured"
            )

        return True


# SearchResult: a plain data container holding one match from a vector
# search, in one shape every backend (Milvus, or any future backend)
# produces the same way. Needed so callers never need to know the raw
# response format of whichever specific database answered the search.
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
        # ".get(name, default)" on a dict reads a key if it exists, or
        # returns "default" instead of raising an error if it does not -
        # safer than "self.fields[field_name]" when a field might be missing.


# VectorDBProvider: the "contract" every vector database backend (like
# MilvusProvider) must follow, plus shared validation logic every backend
# can reuse instead of reimplementing it.
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
        # The "*" makes "embeddings" keyword-only, so subclasses and
        # callers must write "embeddings=..." by name.
        self.config = config
        # self.config: every connection/collection/search setting this
        # provider was built with.
        self.collection_name = config.collection.name
        # self.collection_name: convenience shortcut to the collection name,
        # read often enough to be worth a dedicated attribute.
        self.embedding_dimension = config.collection.embedding_dimension
        # self.embedding_dimension: convenience shortcut used by
        # validate_query_vector below.
        self.embeddings = embeddings
        # self.embeddings: the client that turns search text into vectors,
        # if this provider supports text-based search.
        self.provider_type = (
            self.__class__.__name__.lower().replace("provider", "").replace("vector", "")
        )
        # Turns a class name like "MilvusProvider" into the short label
        # "milvus", used only for the log line below. Two ".replace()"
        # calls are chained to strip both the word "provider" and the
        # word "vector" from the lowercased class name.

        config.validate()
        # Step 1: fail immediately (via VectorDBConfigError) if the config
        # is internally inconsistent, before anything tries to connect.

        logger.info(
            f"Initialized {self.provider_type} vector database provider - "
            f"Collection: {self.collection_name}, "
            f"Dimension: {self.embedding_dimension}"
        )

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the database. Safe to call repeatedly. Raises VectorDBConnectionError."""
        # No body - every subclass must provide its own connection logic.

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
        # "**kwargs" lets each concrete backend accept its own extra,
        # backend-specific search options without this shared interface
        # needing to list every possible one.

    def validate_query_vector(self, query_vector: list[float]) -> bool:
        """Return True if the vector is non-empty and matches the embedding dimension."""
        if not query_vector:
            # Step 1: an empty vector cannot be searched with - reject it
            # early with a clear warning instead of letting the backend
            # fail confusingly.
            logger.warning("Query vector is empty")
            return False

        if len(query_vector) != self.embedding_dimension:
            # Step 2: the vector's length MUST match the collection's
            # configured dimension, or the backend's search call would
            # fail or silently misbehave.
            logger.warning(
                f"Query vector dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(query_vector)}"
            )
            return False

        return True
