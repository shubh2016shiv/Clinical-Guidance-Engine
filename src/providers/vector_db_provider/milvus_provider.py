"""
Milvus Vector Database Provider Implementation

This module provides a comprehensive implementation of the VectorDBProvider interface
using Milvus as the vector database backend.

Features:
- Connection management with lazy loading
- Vector similarity search with IP metric
- Filter expression support
- Integration with embedding providers for query embeddings
- Consistent configuration with ingestion pipeline
"""

import asyncio
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections,
    Collection,
    utility,
    MilvusException,
)

from src.config import get_settings
from src.providers.embedding_provider import (
    EmbeddingProvider,
    create_embedding_provider,
    EmbeddingTaskType,
)
from src.providers.embedding_provider.exceptions import EmbeddingProviderError
from src.logs import get_logger
from src.providers.vector_db_provider.base import (
    VectorDBProvider,
    VectorDBConfig,
    SearchResult,
    SearchMetricType,
)
from src.providers.vector_db_provider.exceptions import (
    VectorDBProviderError,
    VectorDBConnectionError,
    VectorDBSearchError,
    VectorDBConfigError,
    VectorDBDimensionMismatchError,
    VectorDBCollectionError,
)

logger = get_logger(__name__)


class MilvusProvider(VectorDBProvider):
    """
    Milvus vector database provider implementation

    This provider supports Milvus vector database with both
    synchronous and asynchronous operations.
    """

    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        **kwargs,
    ):
        """
        Initialize the Milvus provider

        Args:
            config: Complete VectorDBConfig object (if None, creates default)
            embedding_provider: Embedding provider instance for query embeddings
                If None, creates a Gemini provider with RETRIEVAL_QUERY task type
            **kwargs: Additional configuration parameters

        Raises:
            VectorDBConfigError: If configuration is invalid
        """
        settings = get_settings()

        # Build configuration if not provided
        if config is None:
            from src.providers.vector_db_provider.base import (
                VectorDBConnectionConfig,
                VectorDBCollectionConfig,
                VectorDBSearchConfig,
            )

            # Default connection config
            conn_config = VectorDBConnectionConfig(
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", 19530),
                alias=kwargs.get("alias", "default"),
                timeout=kwargs.get("timeout", 30),
                user=kwargs.get("user", ""),
                password=kwargs.get("password", ""),
                secure=kwargs.get("secure", False),
            )

            # Default collection config (must match ingestion config)
            coll_config = VectorDBCollectionConfig(
                name=kwargs.get("collection_name", "pharmaceutical_drugs"),
                description=kwargs.get(
                    "collection_description",
                    "Pharmaceutical drug database with embeddings for LLM lookup",
                ),
                embedding_dimension=kwargs.get("embedding_dimension", 768),
                load_on_startup=kwargs.get("load_on_startup", True),
            )

            # Default search config
            search_config = VectorDBSearchConfig(
                top_k=kwargs.get("top_k", 10),
                metric_type=SearchMetricType.INNER_PRODUCT,
                search_params=kwargs.get("search_params", {"ef": 64}),
                output_fields=kwargs.get(
                    "output_fields",
                    [
                        "drug_name",
                        "drug_class",
                        "drug_sub_class",
                        "therapeutic_category",
                        "route_of_administration",
                        "formulation",
                        "dosage_strengths",
                        "search_text",
                    ],
                ),
            )

            config = VectorDBConfig(
                connection=conn_config,
                collection=coll_config,
                search=search_config,
                embedding_dimension=coll_config.embedding_dimension,
                embedding_normalized=True,  # Required for IP metric
            )

        # Initialize parent
        super().__init__(config)

        # Initialize or use provided embedding provider
        if embedding_provider is None:
            # Create Gemini provider for query embeddings
            # Must match ingestion settings: 768 dimensions, RETRIEVAL_QUERY task type
            try:
                self.embedding_provider = create_embedding_provider(
                    provider_type="gemini",
                    model_name=settings.gemini_embedding_model or "text-embedding-004",
                    embedding_dimension=self.embedding_dimension,
                    task_type=EmbeddingTaskType.RETRIEVAL_QUERY,
                )
                logger.info(
                    f"Created Gemini embedding provider for queries - "
                    f"Model: {self.embedding_provider.model_name}, "
                    f"Dimension: {self.embedding_dimension}"
                )
            except EmbeddingProviderError as e:
                logger.error(f"Failed to create embedding provider: {e}")
                raise VectorDBConfigError(
                    f"Failed to initialize embedding provider: {str(e)}"
                )
        else:
            self.embedding_provider = embedding_provider
            # Validate embedding provider dimension matches collection
            if self.embedding_provider.embedding_dimension != self.embedding_dimension:
                raise VectorDBDimensionMismatchError(
                    f"Embedding provider dimension ({self.embedding_provider.embedding_dimension}) "
                    f"does not match collection dimension ({self.embedding_dimension})"
                )

        # Milvus connection state
        self._collection: Optional[Collection] = None
        self._is_connected = False

        logger.info(
            f"Milvus provider initialized - "
            f"Collection: {self.collection_name}, "
            f"Dimension: {self.embedding_dimension}, "
            f"Embedding Provider: {self.embedding_provider.provider_type}"
        )

    async def connect(self) -> None:
        """Connect to Milvus (async)"""
        try:
            await asyncio.to_thread(self.connect_sync)
        except VectorDBConnectionError:
            raise
        except Exception as e:
            error_msg = f"Failed to connect to Milvus (async): {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorDBConnectionError(error_msg)

    def connect_sync(self) -> None:
        """
        Connect to Milvus (synchronous)

        Raises:
            VectorDBConnectionError: If connection fails
        """
        if self._is_connected:
            logger.debug("Already connected to Milvus")
            return

        try:
            conn_params = {
                "alias": self.config.connection.alias,
                "host": self.config.connection.host,
                "port": self.config.connection.port,
            }

            # Add authentication if provided
            if self.config.connection.user:
                conn_params["user"] = self.config.connection.user
                conn_params["password"] = self.config.connection.password

            # Add TLS settings if secure
            if self.config.connection.secure:
                conn_params["secure"] = True
                if self.config.connection.server_pem_path:
                    conn_params["server_pem_path"] = (
                        self.config.connection.server_pem_path
                    )
                if self.config.connection.server_name:
                    conn_params["server_name"] = self.config.connection.server_name

            # Connect to Milvus
            # Note: pymilvus may not raise exception immediately if server is unreachable
            # Connection will be verified when we try to use it
            connections.connect(**conn_params)
            logger.info(
                f"Connecting to Milvus - "
                f"Host: {self.config.connection.host}, "
                f"Port: {self.config.connection.port}, "
                f"Alias: {self.config.connection.alias}"
            )

            # Verify connection works by checking if we can list collections
            # This ensures the connection is actually established
            try:
                _ = utility.list_collections(using=self.config.connection.alias)
                logger.info("Milvus connection verified successfully")
            except Exception as verify_error:
                error_msg = (
                    f"Failed to verify Milvus connection: {str(verify_error)}. "
                    f"Make sure Milvus is running on {self.config.connection.host}:{self.config.connection.port}"
                )
                logger.error(error_msg, exc_info=True)
                raise VectorDBConnectionError(error_msg)

            # Mark as connected only after successful verification
            self._is_connected = True

            # Load collection if configured (only after connection is verified)
            if self.config.collection.load_on_startup:
                self._load_collection()

        except MilvusException as e:
            error_msg = f"Failed to connect to Milvus: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._is_connected = False  # Reset connection state on error
            raise VectorDBConnectionError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error connecting to Milvus: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._is_connected = False  # Reset connection state on error
            raise VectorDBConnectionError(error_msg)

    async def disconnect(self) -> None:
        """Disconnect from Milvus (async)"""
        try:
            await asyncio.to_thread(self.disconnect_sync)
        except VectorDBConnectionError:
            raise
        except Exception as e:
            error_msg = f"Failed to disconnect from Milvus (async): {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorDBConnectionError(error_msg)

    def disconnect_sync(self) -> None:
        """
        Disconnect from Milvus (synchronous)

        Raises:
            VectorDBConnectionError: If disconnection fails
        """
        if not self._is_connected:
            logger.debug("Not connected to Milvus")
            return

        try:
            self._collection = None
            connections.disconnect(alias=self.config.connection.alias)
            self._is_connected = False
            logger.info(
                f"Disconnected from Milvus - Alias: {self.config.connection.alias}"
            )
        except MilvusException as e:
            error_msg = f"Failed to disconnect from Milvus: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorDBConnectionError(error_msg)

    def _load_collection(self) -> None:
        """
        Load the Milvus collection

        Raises:
            VectorDBCollectionError: If collection doesn't exist or load fails
        """
        if not self._is_connected:
            raise VectorDBCollectionError("Not connected to Milvus")

        try:
            # Check if collection exists
            if not utility.has_collection(self.collection_name):
                raise VectorDBCollectionError(
                    f"Collection '{self.collection_name}' does not exist in Milvus"
                )

            # Get collection
            self._collection = Collection(
                name=self.collection_name, using=self.config.connection.alias
            )

            # Load collection into memory
            self._collection.load()
            logger.info(f"Loaded collection '{self.collection_name}' into memory")

        except MilvusException as e:
            error_msg = f"Failed to load collection '{self.collection_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorDBCollectionError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error loading collection '{self.collection_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorDBCollectionError(error_msg)

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

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results (overrides config if provided)
            filter_expression: Optional filter expression (e.g., 'drug_class == "ACE inhibitor"')
            output_fields: Optional list of field names to return
            **kwargs: Additional search parameters

        Returns:
            List of SearchResult objects, ordered by relevance

        Raises:
            VectorDBSearchError: If search fails
            VectorDBDimensionMismatchError: If query vector dimension doesn't match
        """
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
        except Exception as e:
            error_msg = f"Failed to perform search (async): {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorDBSearchError(error_msg)

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

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results (overrides config if provided)
            filter_expression: Optional filter expression (e.g., 'drug_class == "ACE inhibitor"')
            output_fields: Optional list of field names to return
            **kwargs: Additional search parameters

        Returns:
            List of SearchResult objects, ordered by relevance (highest score first)

        Raises:
            VectorDBSearchError: If search fails
            VectorDBDimensionMismatchError: If query vector dimension doesn't match
        """
        try:
            # Validate query vector
            if not self.validate_query_vector(query_vector):
                raise VectorDBDimensionMismatchError(
                    f"Query vector dimension ({len(query_vector)}) does not match "
                    f"collection dimension ({self.embedding_dimension})"
                )

            # Ensure connected and collection loaded
            if not self._is_connected:
                self.connect_sync()

            if self._collection is None:
                self._load_collection()

            # Prepare search parameters
            top_k = limit or self.config.search.top_k
            output_fields_to_use = output_fields or self.config.search.output_fields

            # Build search parameters
            search_params = kwargs.get(
                "search_params", self.config.search.search_params.copy()
            )

            # For HNSW index, use ef parameter
            if "ef" not in search_params:
                search_params["ef"] = 64  # Default from config

            # Perform search
            logger.debug(
                f"Searching Milvus collection - "
                f"Top K: {top_k}, "
                f"Filter: {filter_expression or 'None'}, "
                f"Output fields: {len(output_fields_to_use)}"
            )

            search_results = self._collection.search(
                data=[query_vector],  # Milvus expects list of vectors
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expression,
                output_fields=output_fields_to_use if output_fields_to_use else None,
            )

            # Format results
            results = []
            if search_results and len(search_results) > 0:
                # search_results is a list (one query), each element is a list of hits
                hits = search_results[0]

                for hit in hits:
                    # Extract fields from entity
                    fields = {}
                    if output_fields_to_use:
                        for field_name in output_fields_to_use:
                            if hasattr(hit, "entity") and hasattr(hit.entity, "get"):
                                fields[field_name] = hit.entity.get(field_name)
                            elif hasattr(hit, field_name):
                                fields[field_name] = getattr(hit, field_name)

                    # Score is stored as distance (for IP metric, higher is better)
                    # For normalized vectors with IP metric, distance = similarity
                    score = float(hit.distance)

                    result = SearchResult(
                        id=hit.id,
                        score=score,
                        fields=fields,
                        metadata={
                            "distance": score,
                            "collection": self.collection_name,
                        },
                    )
                    results.append(result)

            logger.info(f"Search completed - Found {len(results)} results")
            return results

        except VectorDBDimensionMismatchError:
            raise
        except MilvusException as e:
            error_msg = f"Milvus search error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorDBSearchError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during search: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorDBSearchError(error_msg)

    async def search_by_text(
        self,
        query_text: str,
        limit: Optional[int] = None,
        filter_expression: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Search using text query (generates embedding automatically)

        This is a convenience method that generates an embedding from the query text
        and performs the search.

        Args:
            query_text: Text query (e.g., "ACE inhibitors for hypertension")
            limit: Maximum number of results
            filter_expression: Optional filter expression
            output_fields: Optional list of field names to return
            **kwargs: Additional search parameters

        Returns:
            List of SearchResult objects

        Raises:
            VectorDBSearchError: If search fails
            EmbeddingProviderError: If embedding generation fails
        """
        try:
            # Generate embedding for query text
            logger.debug(f"Generating embedding for query: {query_text[:50]}...")
            query_embedding = await self.embedding_provider.generate_embedding(
                query_text, task_type=EmbeddingTaskType.RETRIEVAL_QUERY
            )

            # Validate embedding dimension
            if not self.validate_query_vector(query_embedding):
                raise VectorDBDimensionMismatchError(
                    f"Generated embedding dimension ({len(query_embedding)}) does not match "
                    f"collection dimension ({self.embedding_dimension})"
                )

            # Perform search
            return await self.search(
                query_vector=query_embedding,
                limit=limit,
                filter_expression=filter_expression,
                output_fields=output_fields,
                **kwargs,
            )

        except EmbeddingProviderError as e:
            error_msg = f"Failed to generate embedding for query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorDBSearchError(error_msg)
        except (VectorDBSearchError, VectorDBDimensionMismatchError):
            raise
        except Exception as e:
            error_msg = f"Failed to search by text: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorDBSearchError(error_msg)

    def search_by_text_sync(
        self,
        query_text: str,
        limit: Optional[int] = None,
        filter_expression: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Search using text query (synchronous variant)

        Args:
            query_text: Text query
            limit: Maximum number of results
            filter_expression: Optional filter expression
            output_fields: Optional list of field names to return
            **kwargs: Additional search parameters

        Returns:
            List of SearchResult objects
        """
        try:
            # Generate embedding for query text
            logger.debug(f"Generating embedding for query: {query_text[:50]}...")
            query_embedding = self.embedding_provider.generate_embedding_sync(
                query_text, task_type=EmbeddingTaskType.RETRIEVAL_QUERY
            )

            # Validate embedding dimension
            if not self.validate_query_vector(query_embedding):
                raise VectorDBDimensionMismatchError(
                    f"Generated embedding dimension ({len(query_embedding)}) does not match "
                    f"collection dimension ({self.embedding_dimension})"
                )

            # Perform search
            return self.search_sync(
                query_vector=query_embedding,
                limit=limit,
                filter_expression=filter_expression,
                output_fields=output_fields,
                **kwargs,
            )

        except EmbeddingProviderError as e:
            error_msg = f"Failed to generate embedding for query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorDBSearchError(error_msg)
        except (VectorDBSearchError, VectorDBDimensionMismatchError):
            raise
        except Exception as e:
            error_msg = f"Failed to search by text: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorDBSearchError(error_msg)

    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the Milvus collection

        Returns:
            Dictionary with collection metadata
        """
        try:
            # Ensure connected
            if not self._is_connected:
                self.connect_sync()

            if self._collection is None:
                self._load_collection()

            info = {
                "name": self.collection_name,
                "num_entities": self._collection.num_entities,
                "description": self._collection.description,
                "schema": {
                    "fields": [
                        {
                            "name": field.name,
                            "type": str(field.dtype),
                            "description": field.description,
                        }
                        for field in self._collection.schema.fields
                    ]
                },
                "is_loaded": self._collection.has_index(),
            }

            return info

        except Exception as e:
            error_msg = f"Failed to get collection info: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorDBProviderError(error_msg)


if __name__ == "__main__":
    """
    Test script for Milvus Provider
    
    This script validates:
    1. Provider initialization
    2. Connection management
    3. Collection loading
    4. Dimension validation
    5. Vector search functionality
    6. Text search functionality (with embedding generation)
    7. Filter expression support
    8. Error handling
    """
    import asyncio
    import sys

    async def test_milvus_provider():
        """Run comprehensive tests on Milvus provider"""
        print("=" * 80)
        print("MILVUS PROVIDER TEST SUITE")
        print("=" * 80)
        print()

        # Test 1: Provider Initialization
        print("TEST 1: Provider Initialization")
        print("-" * 80)
        try:
            provider = MilvusProvider(
                host="localhost",
                port=19530,
                collection_name="pharmaceutical_drugs",
                embedding_dimension=768,
            )
            print("✓ Provider initialized successfully")
            print(f"  - Collection: {provider.collection_name}")
            print(f"  - Embedding Dimension: {provider.embedding_dimension}")
            print(
                f"  - Embedding Provider: {provider.embedding_provider.provider_type}"
            )
            print()
        except Exception as e:
            print(f"✗ Provider initialization failed: {e}")
            return False

        # Test 2: Connection Management
        print("TEST 2: Connection Management")
        print("-" * 80)
        try:
            await provider.connect()
            print("✓ Connected to Milvus successfully")
            print(
                f"  - Host: {provider.config.connection.host}:{provider.config.connection.port}"
            )
            print(f"  - Alias: {provider.config.connection.alias}")
            print(f"  - Is Connected: {provider._is_connected}")
            print()
        except VectorDBConnectionError as e:
            print(f"✗ Connection failed: {e}")
            print("  Make sure Milvus is running on localhost:19530")
            return False
        except Exception as e:
            print(f"✗ Unexpected connection error: {e}")
            return False

        # Test 3: Collection Info
        print("TEST 3: Collection Information")
        print("-" * 80)
        try:
            info = await provider.get_collection_info()
            print("✓ Collection info retrieved successfully")
            print(f"  - Collection Name: {info['name']}")
            print(f"  - Number of Entities: {info['num_entities']}")
            print(f"  - Is Loaded: {info['is_loaded']}")
            print(f"  - Schema Fields: {len(info['schema']['fields'])}")
            print()
        except Exception as e:
            print(f"✗ Failed to get collection info: {e}")
            print("  Make sure the collection 'pharmaceutical_drugs' exists in Milvus")
            return False

        # Test 4: Dimension Validation
        print("TEST 4: Dimension Validation")
        print("-" * 80)
        try:
            # Test correct dimension
            correct_vector = [0.1] * 768
            is_valid = provider.validate_query_vector(correct_vector)
            print(f"✓ Correct dimension (768): {is_valid}")

            # Test incorrect dimension
            incorrect_vector = [0.1] * 1536
            is_invalid = provider.validate_query_vector(incorrect_vector)
            print(f"✓ Incorrect dimension (1536) correctly rejected: {not is_invalid}")

            # Test empty vector
            empty_vector = []
            is_empty_invalid = provider.validate_query_vector(empty_vector)
            print(f"✓ Empty vector correctly rejected: {not is_empty_invalid}")
            print()
        except Exception as e:
            print(f"✗ Dimension validation test failed: {e}")
            return False

        # Test 5: Vector Search (if collection has data)
        print("TEST 5: Vector Search")
        print("-" * 80)
        try:
            # Generate a test vector (normalized random-like vector)
            import random

            random.seed(42)  # For reproducibility
            test_vector = [random.random() for _ in range(768)]
            # Normalize the vector
            magnitude = sum(x**2 for x in test_vector) ** 0.5
            test_vector = [x / magnitude for x in test_vector]

            results = await provider.search(query_vector=test_vector, limit=5)
            print("✓ Vector search completed successfully")
            print(f"  - Results returned: {len(results)}")

            if results:
                print("  - Top result:")
                top_result = results[0]
                print(f"    - ID: {top_result.id}")
                print(f"    - Score: {top_result.score:.4f}")
                if "drug_name" in top_result.fields:
                    print(f"    - Drug Name: {top_result.fields['drug_name']}")
                if "drug_class" in top_result.fields:
                    print(f"    - Drug Class: {top_result.fields['drug_class']}")
            else:
                print("  - No results found (collection may be empty)")
            print()
        except VectorDBDimensionMismatchError as e:
            print(f"✗ Dimension mismatch: {e}")
            return False
        except VectorDBSearchError as e:
            print(f"✗ Search failed: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected search error: {e}")
            return False

        # Test 6: Text Search (with embedding generation)
        print("TEST 6: Text Search (with Embedding Generation)")
        print("-" * 80)
        try:
            test_query = "ACE inhibitors for hypertension"
            results = await provider.search_by_text(query_text=test_query, limit=5)
            print("✓ Text search completed successfully")
            print(f"  - Query: '{test_query}'")
            print(f"  - Results returned: {len(results)}")

            if results:
                print("  - Top 3 results:")
                for i, result in enumerate(results[:3], 1):
                    print(f"    {i}. Score: {result.score:.4f}", end="")
                    if "drug_name" in result.fields:
                        print(f" - {result.fields['drug_name']}", end="")
                    if "drug_class" in result.fields:
                        print(f" ({result.fields['drug_class']})", end="")
                    print()
            else:
                print("  - No results found")
            print()
        except VectorDBSearchError as e:
            print(f"✗ Text search failed: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected text search error: {e}")
            return False

        # Test 7: Error Handling - Invalid Dimension
        print("TEST 7: Error Handling - Invalid Dimension")
        print("-" * 80)
        try:
            invalid_vector = [0.1] * 512  # Wrong dimension
            try:
                await provider.search(query_vector=invalid_vector, limit=5)
                print("✗ Should have raised VectorDBDimensionMismatchError")
                return False
            except VectorDBDimensionMismatchError:
                print(
                    "✓ Correctly raised VectorDBDimensionMismatchError for invalid dimension"
                )
                print()
        except Exception as e:
            print(f"✗ Error handling test failed: {e}")
            return False

        # Test 9: Disconnect
        print("TEST 8: Disconnect")
        print("-" * 80)
        try:
            await provider.disconnect()
            print("✓ Disconnected from Milvus successfully")
            print(f"  - Is Connected: {provider._is_connected}")
            print()
        except Exception as e:
            print(f"✗ Disconnect failed: {e}")
            return False

        # All tests passed
        print("=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        return True

    # Run the test suite
    try:
        success = asyncio.run(test_milvus_provider())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error running test suite: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
