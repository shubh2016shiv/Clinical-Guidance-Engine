"""
Abstract Vector Database Provider Interface

This module provides an abstract interface for vector database providers,
allowing easy extension to support different vector database backends.

The interface supports:
- Vector similarity search
- Filter-based queries
- Both sync and async operations
- Flexible configuration management
- Integration with embedding providers
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from src.logs import get_logger

logger = get_logger(__name__)


class SearchMetricType(Enum):
    """
    Types of similarity metrics for vector search

    Different metrics optimize for different use cases
    """

    COSINE = "cosine"  # Cosine similarity (normalized vectors)
    INNER_PRODUCT = (
        "inner_product"  # Inner product (IP) - works with normalized vectors
    )
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

    def to_dict(self) -> Dict[str, Any]:
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
    search_params: Dict[str, Any] = field(default_factory=dict)

    # Output configuration
    output_fields: List[str] = field(default_factory=list)
    include_metadata: bool = True

    # Filter support
    enable_filters: bool = True

    def to_dict(self) -> Dict[str, Any]:
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

    def to_dict(self) -> Dict[str, Any]:
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
    connection: VectorDBConnectionConfig = field(
        default_factory=VectorDBConnectionConfig
    )

    # Collection configuration
    collection: VectorDBCollectionConfig = field(
        default_factory=VectorDBCollectionConfig
    )

    # Search configuration
    search: VectorDBSearchConfig = field(default_factory=VectorDBSearchConfig)

    # Embedding provider integration
    embedding_dimension: int = 768  # Must match collection embedding_dimension
    embedding_normalized: bool = True  # For IP metric compatibility

    # Additional provider-specific settings
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
            ValueError: If configuration is invalid
        """
        # Validate embedding dimension matches collection dimension
        if self.embedding_dimension != self.collection.embedding_dimension:
            raise ValueError(
                f"Embedding dimension ({self.embedding_dimension}) must match "
                f"collection embedding dimension ({self.collection.embedding_dimension})"
            )

        # Validate metric type is appropriate for normalized embeddings
        if (
            self.embedding_normalized
            and self.search.metric_type != SearchMetricType.INNER_PRODUCT
        ):
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
    fields: Dict[str, Any]  # Field values from the database
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
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
    """
    Abstract base class for vector database providers

    This class defines the interface that all vector database provider implementations
    must follow, ensuring consistency and interchangeability.
    """

    def __init__(self, config: VectorDBConfig):
        """
        Initialize the vector database provider with configuration

        Args:
            config: VectorDBConfig object containing provider settings
        """
        self.config = config
        self.collection_name = config.collection.name
        self.embedding_dimension = config.collection.embedding_dimension
        self.provider_type = (
            self.__class__.__name__.lower()
            .replace("provider", "")
            .replace("vector", "")
        )

        # Validate configuration
        try:
            config.validate()
        except ValueError as e:
            logger.error(f"Invalid vector database provider configuration: {e}")
            raise

        logger.info(
            f"Initialized {self.provider_type} vector database provider - "
            f"Collection: {self.collection_name}, "
            f"Dimension: {self.embedding_dimension}"
        )

    @abstractmethod
    async def connect(self) -> None:
        """
        Connect to the vector database (async)

        This method establishes a connection to the vector database.
        Should be idempotent (safe to call multiple times).

        Raises:
            VectorDBConnectionError: If connection fails
        """
        pass

    @abstractmethod
    def connect_sync(self) -> None:
        """
        Connect to the vector database (synchronous)

        This method establishes a connection to the vector database.
        Should be idempotent (safe to call multiple times).

        Raises:
            VectorDBConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the vector database (async)

        This method closes the connection and cleans up resources.

        Raises:
            VectorDBConnectionError: If disconnection fails
        """
        pass

    @abstractmethod
    def disconnect_sync(self) -> None:
        """
        Disconnect from the vector database (synchronous)

        This method closes the connection and cleans up resources.

        Raises:
            VectorDBConnectionError: If disconnection fails
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        limit: Optional[int] = None,
        filter_expression: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Perform vector similarity search (async)

        This is the primary method for searching the vector database.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results (overrides config if provided)
            filter_expression: Optional filter expression (provider-specific syntax)
            output_fields: Optional list of field names to return
            **kwargs: Additional provider-specific parameters

        Returns:
            List of SearchResult objects, ordered by relevance (highest score first)

        Raises:
            VectorDBSearchError: If search fails
            VectorDBDimensionMismatchError: If query vector dimension doesn't match
        """
        pass

    @abstractmethod
    def search_sync(
        self,
        query_vector: List[float],
        limit: Optional[int] = None,
        filter_expression: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Perform vector similarity search (synchronous)

        This method is useful for non-async contexts or when running
        in synchronous code.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results (overrides config if provided)
            filter_expression: Optional filter expression (provider-specific syntax)
            output_fields: Optional list of field names to return
            **kwargs: Additional provider-specific parameters

        Returns:
            List of SearchResult objects, ordered by relevance (highest score first)

        Raises:
            VectorDBSearchError: If search fails
            VectorDBDimensionMismatchError: If query vector dimension doesn't match
        """
        pass

    @abstractmethod
    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection

        Returns:
            Dictionary with collection metadata (name, entity count, schema, etc.)

        Raises:
            VectorDBError: If operation fails
        """
        pass

    def validate_query_vector(self, query_vector: List[float]) -> bool:
        """
        Validate that a query vector has the expected dimension

        Args:
            query_vector: Query embedding vector to validate

        Returns:
            True if valid, False otherwise
        """
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

    def update_config(self, **kwargs) -> None:
        """
        Update provider configuration dynamically

        This allows runtime modification of settings without recreating
        the provider instance.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Attempted to set unknown config key: {key}")

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the vector database provider

        Returns:
            Dictionary with provider information and current configuration
        """
        return {
            "provider_type": self.provider_type,
            "collection_name": self.collection_name,
            "embedding_dimension": self.embedding_dimension,
            "config": self.config.to_dict(),
        }
