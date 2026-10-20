"""
Vector Database Ingestion Module

Handles embedding generation and ingestion of pharmaceutical data into Milvus.
Implements robust error handling, batch processing, and progress tracking.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from config_loader import IngestionConfig, MilvusConfig, OpenAIConfig
from drugs_etl_pipeline import StandardizedDrugRecord
from openai import OpenAI
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)

logger = logging.getLogger(__name__)


class OpenAIEmbeddingGenerator:
    """Generate unit-length document embeddings with the OpenAI Embeddings API."""

    def __init__(self, config: OpenAIConfig):
        self.config = config
        # The SDK retries rate limits and transient errors with backoff.
        self.client = OpenAI(
            api_key=config.api_key,
            max_retries=config.retry_attempts,
            timeout=config.timeout_seconds,
        )

    def generate_embeddings_batch(
        self, texts: list[str], show_progress: bool = True
    ) -> list[np.ndarray]:
        """
        Return one unit-length embedding per text, in order.

        Raises ValueError for a blank text or an unexpected vector size. API errors
        propagate once the SDK's retries are exhausted, so a failed run never
        inserts partial or zero vectors.
        """
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed a blank search text.")

        starts = range(0, len(texts), self.config.batch_size)
        if show_progress and HAS_TQDM:
            starts = tqdm(starts, desc="Embedding batches")

        embeddings: list[np.ndarray] = []
        for start in starts:
            batch = texts[start : start + self.config.batch_size]
            response = self.client.embeddings.create(
                model=self.config.model_name,
                input=batch,
                dimensions=self.config.dimensions,
            )
            for item in sorted(response.data, key=lambda item: item.index):
                vector = np.array(item.embedding, dtype=np.float32)
                if vector.shape[0] != self.config.dimensions:
                    raise ValueError(
                        f"Expected {self.config.dimensions}-dimensional embeddings "
                        f"from {self.config.model_name}, got {vector.shape[0]}."
                    )
                embeddings.append(vector / np.linalg.norm(vector))

        return embeddings


class MilvusCollectionManager:
    """
    Manage Milvus collection lifecycle: connection, schema, index, and operations.
    """

    def __init__(self, config: MilvusConfig):
        self.config = config
        self.collection: Collection | None = None
        self.is_connected = False

    def connect(self):
        """Establish connection to Milvus server"""
        try:
            conn_params = {
                "alias": self.config.connection.alias,
                "host": self.config.connection.host,
                "port": self.config.connection.port,
                "timeout": self.config.connection.timeout,
            }

            if self.config.connection.user:
                conn_params["user"] = self.config.connection.user
                conn_params["password"] = self.config.connection.password

            if self.config.connection.secure:
                conn_params["secure"] = True
                if self.config.connection.server_pem_path:
                    conn_params["server_pem_path"] = self.config.connection.server_pem_path
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
                description="OpenAI embedding vector",
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
            drop_if_exists if drop_if_exists is not None else self.config.collection.drop_existing
        )

        # Every call below must name the alias, or it silently acts on the
        # "default" connection instead of the one connect() opened.
        alias = self.config.connection.alias

        if utility.has_collection(collection_name, using=alias):
            if should_drop:
                utility.drop_collection(collection_name, using=alias)
            else:
                # Reuse the existing collection, but warn if its vector width differs
                existing_collection = Collection(collection_name, using=alias)
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
            raise RuntimeError("Collection not initialized. Call create_collection() first.")

        index_params = {
            "index_type": self.config.index.type,
            "metric_type": self.config.index.metric_type,
            "params": self.config.index.params,
        }

        try:
            self.collection.create_index(field_name="embedding", index_params=index_params)

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
        records: list[StandardizedDrugRecord],
        embeddings: list[np.ndarray],
        search_texts: list[str],
        batch_size: int | None = None,
    ) -> int:
        """Insert records, their embeddings and the search text each was embedded from.

        ``search_texts`` is passed in rather than regenerated so that the stored
        search_text column is exactly the string that produced the stored vector.
        """
        if self.collection is None:
            raise RuntimeError("Collection not initialized")

        if not (len(records) == len(embeddings) == len(search_texts)):
            raise ValueError(
                f"Counts must match: {len(records)} records, {len(embeddings)} "
                f"embeddings, {len(search_texts)} search texts"
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
            batch_search_texts = search_texts[batch_start:batch_end]

            data = self._prepare_batch_data(batch_records, batch_embeddings, batch_search_texts)

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
        self,
        records: list[StandardizedDrugRecord],
        embeddings: list[np.ndarray],
        search_texts: list[str],
    ) -> list[list[Any]]:
        """Prepare batch data in column format for Milvus insertion.

        Column order must match the field order in _create_collection_schema():
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
        if not (len(records) == len(embeddings) == len(search_texts)):
            raise ValueError(
                f"Cannot prepare batch data: {len(records)} records, "
                f"{len(embeddings)} embeddings, {len(search_texts)} search texts"
            )

        # Validate all embeddings are numpy arrays with correct shape
        # Use the actual dimension from the embeddings themselves
        if embeddings:
            actual_dim = embeddings[0].shape[0]
            expected_dim = self.config.collection.embedding_dimension

            # Check if there's a dimension mismatch with collection config
            if actual_dim != expected_dim:
                error_msg = (
                    f"\n{'=' * 70}\n"
                    f"COLLECTION DIMENSION MISMATCH\n"
                    f"{'=' * 70}\n"
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
                    f"{'=' * 70}\n"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Validate each embedding
            for i, emb in enumerate(embeddings):
                if not isinstance(emb, np.ndarray):
                    raise TypeError(f"Embedding at index {i} is not a numpy array: {type(emb)}")
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
            list(search_texts),
            [emb.tolist() for emb in embeddings],
        ]

        # Validate all columns have same length
        lengths = [len(col) for col in data]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Column length mismatch in batch data preparation: {dict(zip(range(len(lengths)), lengths, strict=False))}"
            )

        return data

    def get_collection_stats(self) -> dict[str, Any]:
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

        # Validate dimension consistency between OpenAI and Milvus configs
        if config.openai.dimensions != config.milvus.collection.embedding_dimension:
            raise ValueError(
                f"Dimension mismatch: OpenAI embedding dimensions ({config.openai.dimensions}) "
                f"must match Milvus collection embedding_dimension ({config.milvus.collection.embedding_dimension})"
            )

        self.embedding_generator = OpenAIEmbeddingGenerator(config.openai)
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

    def run(self, records: list[StandardizedDrugRecord]) -> dict[str, Any]:
        """Embed the records and insert them into Milvus.

        Raises:
            ValueError: If there are no records, rather than reporting a successful
                run that ingested nothing.
        """
        if not records:
            raise ValueError(
                "No records to ingest. Run the ETL pipeline first and check "
                "data_sources.input_files in ingestion_config.yaml."
            )

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

            # Step 3: Generate embeddings. These strings are rendered once here and
            # reused for the stored search_text column, so the two cannot diverge.
            print("Step 3: Generating embeddings")
            search_texts = [
                record.generate_search_text(self.config.search_text) for record in records
            ]

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
                records, embeddings, search_texts
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
            self.stats["duration_seconds"] = self.stats["end_time"] - self.stats["start_time"]
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
            print(f"  Total entities: {self.stats['collection_stats'].get('num_entities')}")

        print("=" * 60)


# Example usage and testing
if __name__ == "__main__":
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent))

    from config_loader import ConfigLoader
    from drugs_etl_pipeline import PharmaceuticalETLPipeline

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
