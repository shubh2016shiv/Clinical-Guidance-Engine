"""
Vector Database Ingestion Module

Handles embedding generation and ingestion of pharmaceutical data into Milvus.
Implements robust error handling, batch processing, and progress tracking.
"""

import time
import json
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import sys

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import google.generativeai as genai
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException,
)

from drugs_etl_pipeline import StandardizedDrugRecord
from config_loader import IngestionConfig, GeminiConfig, MilvusConfig

logger = logging.getLogger(__name__)


class GeminiEmbeddingGenerator:
    """
    Generate embeddings using Google Gemini API.
    Handles batch processing, rate limiting, retries, and error recovery.
    """

    def __init__(self, config: GeminiConfig):
        self.config = config
        self._configure_api()
        self._initialize_rate_limiter()
        self._actual_dimension = None  # Will be set on first embedding generation
        self._dimension_validated = False

    def _configure_api(self):
        """Configure Gemini API with credentials"""
        try:
            genai.configure(api_key=self.config.api_key)
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            raise

    def _initialize_rate_limiter(self):
        """Initialize rate limiting parameters"""
        self.requests_interval = 1.0 / self.config.requests_per_second
        self.last_request_time = 0.0

    def _apply_rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.requests_interval:
            sleep_time = self.requests_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _validate_and_set_dimension(self, embedding: np.ndarray):
        """
        Validate embedding dimension on first call and set actual dimension.
        Raises error if dimension doesn't match config.
        """
        if self._dimension_validated:
            return

        actual_dim = embedding.shape[0]
        expected_dim = self.config.embedding.dimensions

        if actual_dim != expected_dim:
            error_msg = (
                f"\n{'='*70}\n"
                f"EMBEDDING DIMENSION MISMATCH\n"
                f"{'='*70}\n"
                f"Expected dimension: {expected_dim} (from config)\n"
                f"Actual dimension:   {actual_dim} (from Gemini API)\n"
                f"\n"
                f"The Gemini API returned embeddings with {actual_dim} dimensions,\n"
                f"but your configuration specifies {expected_dim} dimensions.\n"
                f"\n"
                f"SOLUTION:\n"
                f"Update 'gemini.embedding.dimensions' in your config file to {actual_dim}\n"
                f"AND update 'milvus.collection.embedding_dimension' to {actual_dim}\n"
                f"\n"
                f"Config file: ingestion_config.yaml\n"
                f"{'='*70}\n"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        self._actual_dimension = actual_dim
        self._dimension_validated = True
        logger.info(f"Embedding dimension validated: {actual_dim}")

    def get_actual_dimension(self) -> Optional[int]:
        """Return the actual embedding dimension after validation."""
        return self._actual_dimension

    def generate_embedding(
        self, text: str, retry_count: int = 0
    ) -> Optional[np.ndarray]:
        """
        Generate embedding for single text.
        """
        if not text or not text.strip():
            # Return zero vector with actual dimension if validated, else config dimension
            dim = (
                self._actual_dimension
                if self._actual_dimension
                else self.config.embedding.dimensions
            )
            return np.zeros(dim, dtype=np.float32)

        try:
            self._apply_rate_limit()

            result = genai.embed_content(
                model=self.config.model_name,
                content=text,
                task_type=self.config.task_type,
                output_dimensionality=self.config.embedding.dimensions,
            )

            embedding = np.array(result["embedding"], dtype=np.float32)

            # Validate dimension on first successful call
            if not self._dimension_validated:
                self._validate_and_set_dimension(embedding)

            # Normalize if configured
            if self.config.embedding.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:  # Avoid division by zero
                    embedding = embedding / norm

            return embedding

        except Exception as e:
            # Retry logic
            if retry_count < self.config.retry_attempts:
                delay = self.config.retry_delay_seconds
                if self.config.exponential_backoff:
                    delay *= 2**retry_count

                time.sleep(delay)
                return self.generate_embedding(text, retry_count + 1)

            if self.config.skip_on_error:
                logger.warning(
                    f"Skipping embedding for text after {retry_count} retries: {e}"
                )
                dim = (
                    self._actual_dimension
                    if self._actual_dimension
                    else self.config.embedding.dimensions
                )
                return np.zeros(dim, dtype=np.float32)

            raise

    def generate_embeddings_batch(
        self, texts: List[str], show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches.
        Ensures exactly one embedding per input text with proper error handling.
        """
        if not texts:
            return []

        embeddings = []

        for batch_idx in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[batch_idx : batch_idx + self.config.batch_size]
            batch_embeddings = []

            try:
                self._apply_rate_limit()

                result = genai.embed_content(
                    model=self.config.model_name,
                    content=batch_texts,
                    task_type=self.config.task_type,
                    output_dimensionality=self.config.embedding.dimensions,
                )

                batch_embeddings_raw = result["embedding"]

                # Validate batch response
                if len(batch_embeddings_raw) != len(batch_texts):
                    logger.warning(
                        f"Batch API returned {len(batch_embeddings_raw)} embeddings "
                        f"for {len(batch_texts)} texts. Processing individually."
                    )
                    # Discard partial batch response and process all texts individually
                    batch_embeddings = self._process_texts_individually(batch_texts)
                else:
                    # Convert and normalize embeddings
                    for i, embedding in enumerate(batch_embeddings_raw):
                        emb_array = np.array(embedding, dtype=np.float32)

                        # Validate dimension on first successful embedding
                        if not self._dimension_validated and i == 0:
                            self._validate_and_set_dimension(emb_array)

                        if self.config.embedding.normalize:
                            norm = np.linalg.norm(emb_array)
                            if norm > 0:
                                emb_array = emb_array / norm

                        batch_embeddings.append(emb_array)

            except Exception as e:
                logger.warning(
                    f"Batch embedding failed at index {batch_idx}: {e}. Processing individually."
                )
                # Process entire batch individually
                batch_embeddings = self._process_texts_individually(batch_texts)

            # Validate we got embeddings for all texts in this batch
            if len(batch_embeddings) != len(batch_texts):
                raise RuntimeError(
                    f"Critical error: Generated {len(batch_embeddings)} embeddings "
                    f"for {len(batch_texts)} texts in batch starting at index {batch_idx}"
                )

            embeddings.extend(batch_embeddings)

        # Final validation
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Embedding generation failed: generated {len(embeddings)} embeddings "
                f"for {len(texts)} texts"
            )

        return embeddings

    def _process_texts_individually(self, texts: List[str]) -> List[np.ndarray]:
        """
        Process texts one by one with proper error handling.
        Guarantees one embedding per text.
        """
        embeddings = []

        for text in texts:
            embedding = self.generate_embedding(text)

            if embedding is None:
                # generate_embedding returns None only if skip_on_error is False
                # and all retries failed - this should raise an exception
                logger.error(f"Failed to generate embedding for text: {text[:100]}...")
                dim = (
                    self._actual_dimension
                    if self._actual_dimension
                    else self.config.embedding.dimensions
                )
                embedding = np.zeros(dim, dtype=np.float32)

            embeddings.append(embedding)

        return embeddings


class MilvusCollectionManager:
    """
    Manage Milvus collection lifecycle: connection, schema, index, and operations.
    """

    def __init__(self, config: MilvusConfig):
        self.config = config
        self.collection: Optional[Collection] = None
        self.is_connected = False

    def connect(self):
        """Establish connection to Milvus server"""
        try:
            conn_params = {
                "alias": self.config.connection.alias,
                "host": self.config.connection.host,
                "port": self.config.connection.port,
            }

            if self.config.connection.user:
                conn_params["user"] = self.config.connection.user
                conn_params["password"] = self.config.connection.password

            if self.config.connection.secure:
                conn_params["secure"] = True
                if self.config.connection.server_pem_path:
                    conn_params["server_pem_path"] = (
                        self.config.connection.server_pem_path
                    )
                if self.config.connection.server_name:
                    conn_params["server_name"] = self.config.connection.server_name

            connections.connect(**conn_params)
            self.is_connected = True

        except MilvusException as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def disconnect(self):
        """Disconnect from Milvus server"""
        if self.is_connected:
            connections.disconnect(alias=self.config.connection.alias)
            self.is_connected = False

    def _create_collection_schema(self) -> CollectionSchema:
        """Create collection schema for pharmaceutical drugs."""
        fields = [
            # Primary key
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="Auto-generated primary key",
            ),
            # Drug identification
            FieldSchema(
                name="drug_name",
                dtype=DataType.VARCHAR,
                max_length=500,
                description="Original drug name",
            ),
            # Drug classification
            FieldSchema(
                name="drug_class",
                dtype=DataType.VARCHAR,
                max_length=200,
                description="Drug class",
            ),
            FieldSchema(
                name="drug_sub_class",
                dtype=DataType.VARCHAR,
                max_length=200,
                description="Drug sub-class",
            ),
            FieldSchema(
                name="therapeutic_category",
                dtype=DataType.VARCHAR,
                max_length=200,
                description="Therapeutic category",
            ),
            # Drug administration
            FieldSchema(
                name="route_of_administration",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="Route of administration",
            ),
            FieldSchema(
                name="formulation",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="Drug formulation",
            ),
            # Available Dosage Strengths for the drug
            FieldSchema(
                name="dosage_strengths",
                dtype=DataType.VARCHAR,
                max_length=2000,
                description="Available Dosage Strengths for the drug",
            ),
            # Search text
            FieldSchema(
                name="search_text",
                dtype=DataType.VARCHAR,
                max_length=2000,
                description="Combined searchable text",
            ),
            # Embedding vector
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.config.collection.embedding_dimension,
                description="Gemini embedding vector",
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description=self.config.collection.description,
            enable_dynamic_field=False,
        )

        return schema

    def create_collection(self, drop_if_exists: bool = None):
        """Create collection with defined schema."""
        collection_name = self.config.collection.name
        should_drop = (
            drop_if_exists
            if drop_if_exists is not None
            else self.config.collection.drop_existing
        )

        # Check if collection exists
        if utility.has_collection(collection_name):
            if should_drop:
                utility.drop_collection(collection_name)
            else:
                # Check if existing collection has matching dimensions
                existing_collection = Collection(collection_name)
                self.collection = existing_collection

                # Validate embedding dimension matches config
                try:
                    # Get embedding field from schema
                    for field in existing_collection.schema.fields:
                        if field.name == "embedding":
                            # Extract dimension from field (FieldSchema for FLOAT_VECTOR has 'params' with 'dim')
                            existing_dim = None
                            if (
                                hasattr(field, "params")
                                and isinstance(field.params, dict)
                                and "dim" in field.params
                            ):
                                existing_dim = field.params["dim"]
                            elif hasattr(field, "dim"):
                                # Direct attribute access
                                existing_dim = field.dim

                            if existing_dim is not None:
                                config_dim = self.config.collection.embedding_dimension

                                if existing_dim != config_dim:
                                    warning_msg = (
                                        f"WARNING: Existing collection '{collection_name}' has embedding dimension {existing_dim}, "
                                        f"but config specifies {config_dim}. The collection will NOT be recreated because "
                                        f"drop_existing=false. To fix this, either:\n"
                                        f"  1. Set drop_existing=true in config to recreate the collection, or\n"
                                        f"  2. Manually drop the collection in Milvus before running the pipeline.\n"
                                        f"Current embeddings will have {config_dim} dimensions and cannot be inserted into a "
                                        f"collection expecting {existing_dim} dimensions."
                                    )
                                    logger.warning(warning_msg)
                                    print(f"\n{warning_msg}\n")
                                    break
                except (AttributeError, KeyError, TypeError) as e:
                    # If we can't determine the dimension, log a warning but continue
                    logger.debug(f"Could not validate collection dimensions: {e}")

                return

        # Create new collection
        schema = self._create_collection_schema()

        try:
            self.collection = Collection(
                name=collection_name, schema=schema, using=self.config.connection.alias
            )

        except MilvusException as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    def create_index(self):
        """Create index on embedding field for efficient similarity search"""
        if self.collection is None:
            raise RuntimeError(
                "Collection not initialized. Call create_collection() first."
            )

        index_params = {
            "index_type": self.config.index.type,
            "metric_type": self.config.index.metric_type,
            "params": self.config.index.params,
        }

        try:
            self.collection.create_index(
                field_name="embedding", index_params=index_params
            )

        except MilvusException as e:
            logger.error(f"Failed to create index: {e}")
            raise

    def load_collection(self):
        """Load collection into memory for searching"""
        if self.collection is None:
            raise RuntimeError("Collection not initialized")

        try:
            self.collection.load()
        except MilvusException as e:
            logger.error(f"Failed to load collection: {e}")
            raise

    def insert_records(
        self,
        records: List[StandardizedDrugRecord],
        embeddings: List[np.ndarray],
        batch_size: int = None,
    ) -> int:
        """
        Insert drug records with embeddings into collection.
        """
        if self.collection is None:
            raise RuntimeError("Collection not initialized")

        if len(records) != len(embeddings):
            raise ValueError(
                f"Records ({len(records)}) and embeddings ({len(embeddings)}) count mismatch"
            )

        batch_size = batch_size or self.config.insertion.batch_size
        total_inserted = 0

        # Use tqdm for progress bar if available
        if HAS_TQDM:
            iterator = tqdm(
                range(0, len(records), batch_size),
                desc="Inserting records",
                unit="batch",
            )
        else:
            iterator = range(0, len(records), batch_size)

        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(records))
            batch_records = records[batch_start:batch_end]
            batch_embeddings = embeddings[batch_start:batch_end]

            # Prepare data in column format
            data = self._prepare_batch_data(batch_records, batch_embeddings)

            try:
                # Insert batch
                self.collection.insert(data)

                total_inserted += len(batch_records)

            except MilvusException as e:
                logger.error(f"Failed to insert batch {batch_start}-{batch_end}: {e}")
                raise

        # Flush all pending inserts at the end
        self.collection.flush()
        return total_inserted

    def _prepare_batch_data(
        self, records: List[StandardizedDrugRecord], embeddings: List[np.ndarray]
    ) -> List[List[Any]]:
        """Prepare batch data in column format for Milvus insertion.

        Column order must match schema in ingestion_config.yaml:
        1. drug_name
        2. drug_class
        3. drug_sub_class
        4. therapeutic_category
        5. route_of_administration
        6. formulation
        7. dosage_strengths
        8. search_text
        9. embedding
        """
        # Validate inputs
        if len(records) != len(embeddings):
            raise ValueError(
                f"Cannot prepare batch data: {len(records)} records but {len(embeddings)} embeddings"
            )

        # Validate all embeddings are numpy arrays with correct shape
        # Use the actual dimension from the embeddings themselves
        if embeddings:
            actual_dim = embeddings[0].shape[0]
            expected_dim = self.config.collection.embedding_dimension

            # Check if there's a dimension mismatch with collection config
            if actual_dim != expected_dim:
                error_msg = (
                    f"\n{'='*70}\n"
                    f"COLLECTION DIMENSION MISMATCH\n"
                    f"{'='*70}\n"
                    f"Embedding dimension: {actual_dim}\n"
                    f"Collection expects: {expected_dim}\n"
                    f"\n"
                    f"The embeddings have {actual_dim} dimensions, but the Milvus\n"
                    f"collection is configured for {expected_dim} dimensions.\n"
                    f"\n"
                    f"SOLUTION:\n"
                    f"Update 'milvus.collection.embedding_dimension' in your config\n"
                    f"file to {actual_dim}, then set 'milvus.collection.drop_existing'\n"
                    f"to true to recreate the collection.\n"
                    f"\n"
                    f"Config file: ingestion_config.yaml\n"
                    f"{'='*70}\n"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Validate each embedding
            for i, emb in enumerate(embeddings):
                if not isinstance(emb, np.ndarray):
                    raise TypeError(
                        f"Embedding at index {i} is not a numpy array: {type(emb)}"
                    )
                if emb.shape[0] != actual_dim:
                    raise ValueError(
                        f"Embedding at index {i} has dimension {emb.shape[0]}, expected {actual_dim}"
                    )

        data = [
            [r.drug_name for r in records],
            [r.drug_class or "" for r in records],
            [r.drug_sub_class or "" for r in records],
            [r.therapeutic_category or "" for r in records],
            [r.route_of_administration or "" for r in records],
            [r.formulation or "" for r in records],
            [json.dumps(r.dosages) for r in records],
            [r.generate_search_text() for r in records],
            [emb.tolist() for emb in embeddings],
        ]

        # Validate all columns have same length
        lengths = [len(col) for col in data]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Column length mismatch in batch data preparation: {dict(zip(range(len(lengths)), lengths))}"
            )

        return data

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if self.collection is None:
            return {}

        try:
            stats = {
                "name": self.collection.name,
                "num_entities": self.collection.num_entities,
                "schema": {
                    "description": self.collection.description,
                    "fields": len(self.collection.schema.fields),
                },
            }
            return stats
        except MilvusException as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}


class VectorDBIngestionPipeline:
    """
    Complete vector database ingestion pipeline.
    Orchestrates ETL processing, embedding generation, and Milvus insertion.
    """

    def __init__(self, config: IngestionConfig):
        self.config = config

        # Validate dimension consistency between Gemini and Milvus configs
        if (
            config.gemini.embedding.dimensions
            != config.milvus.collection.embedding_dimension
        ):
            raise ValueError(
                f"Dimension mismatch: Gemini embedding dimensions ({config.gemini.embedding.dimensions}) "
                f"must match Milvus collection embedding_dimension ({config.milvus.collection.embedding_dimension})"
            )

        self.embedding_generator = GeminiEmbeddingGenerator(config.gemini)
        self.collection_manager = MilvusCollectionManager(config.milvus)

        # Statistics
        self.stats = {
            "total_records": 0,
            "embeddings_generated": 0,
            "records_inserted": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None,
        }

    def run(self, records: List[StandardizedDrugRecord]) -> Dict[str, Any]:
        """
        Execute complete ingestion pipeline.
        """
        self.stats["start_time"] = time.time()
        self.stats["total_records"] = len(records)

        print("=" * 60)
        print("VECTOR DATABASE INGESTION STARTED")
        print("=" * 60)

        try:
            # Step 1: Connect to Milvus
            print("Step 1: Connecting to Milvus")
            self.collection_manager.connect()

            # Step 2: Create/setup collection
            print("Step 2: Setting up collection")
            self.collection_manager.create_collection()

            # Step 3: Generate embeddings
            print("Step 3: Generating embeddings")
            search_texts = [record.generate_search_text() for record in records]

            # Generate all embeddings at once - the method handles batching internally
            embeddings = self.embedding_generator.generate_embeddings_batch(
                search_texts, show_progress=True
            )

            self.stats["embeddings_generated"] = len(embeddings)

            # Validate embedding count matches record count
            if len(embeddings) != len(records):
                raise ValueError(
                    f"Embedding count ({len(embeddings)}) does not match record count ({len(records)}). "
                    f"This indicates a bug in embedding generation."
                )

            # Step 4: Insert into Milvus
            print("Step 4: Inserting records")
            records_inserted = self.collection_manager.insert_records(
                records, embeddings
            )
            self.stats["records_inserted"] = records_inserted

            # Step 5: Create index
            print("Step 5: Creating index")
            self.collection_manager.create_index()

            # Step 6: Load collection
            if self.config.milvus.performance.load_on_startup:
                print("Step 6: Loading collection")
                self.collection_manager.load_collection()

            # Step 7: Verify and get stats
            print("Step 7: Verifying ingestion")
            collection_stats = self.collection_manager.get_collection_stats()

            self.stats["end_time"] = time.time()
            self.stats["duration_seconds"] = (
                self.stats["end_time"] - self.stats["start_time"]
            )
            self.stats["collection_stats"] = collection_stats

            self._print_final_report()
            return self.stats

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.stats["errors"] += 1
            raise

        finally:
            # Cleanup
            if self.collection_manager.is_connected:
                self.collection_manager.disconnect()

    def _print_final_report(self):
        """Print final pipeline execution report"""
        print("\n" + "=" * 60)
        print("INGESTION COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print("Statistics:")
        print(f"  Records processed: {self.stats['total_records']}")
        print(f"  Records inserted: {self.stats['records_inserted']}")
        print(f"  Duration: {self.stats['duration_seconds']:.2f}s")

        if self.stats.get("collection_stats"):
            print(f"  Collection: {self.stats['collection_stats'].get('name')}")
            print(
                f"  Total entities: {self.stats['collection_stats'].get('num_entities')}"
            )

        print("=" * 60)


# Example usage and testing
if __name__ == "__main__":
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent))

    from drugs_etl_pipeline import PharmaceuticalETLPipeline
    from config_loader import ConfigLoader

    # Setup minimal logging - only errors
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

    try:
        # Load configuration
        config_loader = ConfigLoader("ingestion_config.yaml")
        config = config_loader.load()

        # Run ETL pipeline first with config
        etl_pipeline = PharmaceuticalETLPipeline(config=config)
        records = etl_pipeline.run_pipeline(config.data_sources.input_files)

        # Save intermediate standardized CSV file for inspection
        base_path = Path(__file__).parent.parent
        output_dir = base_path / config.data_sources.output_directory
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / config.data_sources.standardized_data_file)
        etl_pipeline.save_to_csv(output_path)
        print(f"Intermediate standardized data saved to: {output_path}")

        # Run vector DB ingestion
        ingestion_pipeline = VectorDBIngestionPipeline(config)
        stats = ingestion_pipeline.run(records)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
