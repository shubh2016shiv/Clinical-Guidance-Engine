"""Exercise the Milvus ingestion pipeline against a mocked pymilvus, with no server."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent.parent / "drug_data_ingestion_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from config_loader import (  # noqa: E402
    IngestionConfig,
    MilvusCollectionConfig,
    MilvusConfig,
    MilvusConnectionConfig,
    OpenAIConfig,
    SearchTextConfig,
)
from drug_ingestion_pipeline import (  # noqa: E402
    MilvusCollectionManager,
    VectorDBIngestionPipeline,
)
from drugs_etl_pipeline import StandardizedDrugRecord  # noqa: E402

MODULE = "drug_ingestion_pipeline"
DIMENSION = 4


def milvus_config(alias="default"):
    return MilvusConfig(
        connection=MilvusConnectionConfig(alias=alias),
        collection=MilvusCollectionConfig(embedding_dimension=DIMENSION),
    )


def records(count=2):
    return [
        StandardizedDrugRecord(
            drug_name=f"Drug {i}",
            base_drug_name=f"Drug {i}",
            drug_class="Antidiabetic",
            dosages=[f"{i}00 mg"],
            route_of_administration="oral",
            formulation="tablet",
        )
        for i in range(count)
    ]


def embeddings(count=2):
    return [np.ones(DIMENSION, dtype=np.float32) for _ in range(count)]


class BatchDataTests(unittest.TestCase):
    """The column block handed to Milvus."""

    def setUp(self) -> None:
        self.manager = MilvusCollectionManager(milvus_config())

    def test_columns_match_the_schema_order_and_have_equal_length(self) -> None:
        data = self.manager._prepare_batch_data(records(), embeddings(), ["text 0", "text 1"])

        self.assertEqual(len(data), 9)
        self.assertEqual({len(column) for column in data}, {2})
        schema_names = [
            field.name
            for field in self.manager._create_collection_schema().fields
            if field.name != "id"
        ]
        self.assertEqual(len(schema_names), len(data))

    def test_the_stored_search_text_is_the_one_passed_in(self) -> None:
        """It must be the string that produced the vector, not a fresh render."""
        data = self.manager._prepare_batch_data(
            records(), embeddings(), ["embedded A", "embedded B"]
        )

        self.assertEqual(data[7], ["embedded A", "embedded B"])

    def test_a_count_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.manager._prepare_batch_data(records(2), embeddings(2), ["only one"])

    def test_an_embedding_of_the_wrong_width_is_rejected(self) -> None:
        wrong = [np.ones(DIMENSION + 1, dtype=np.float32) for _ in range(2)]

        with self.assertRaises(ValueError) as caught:
            self.manager._prepare_batch_data(records(), wrong, ["a", "b"])

        self.assertIn("DIMENSION MISMATCH", str(caught.exception))


class CollectionAliasTests(unittest.TestCase):
    """Every collection-level call must name the alias connect() opened."""

    def test_create_collection_passes_the_configured_alias(self) -> None:
        manager = MilvusCollectionManager(milvus_config(alias="asclepius"))

        with (
            patch(f"{MODULE}.utility") as utility,
            patch(f"{MODULE}.Collection") as collection,
        ):
            utility.has_collection.return_value = False
            manager.create_collection()

        self.assertEqual(utility.has_collection.call_args.kwargs["using"], "asclepius")
        self.assertEqual(collection.call_args.kwargs["using"], "asclepius")

    def test_dropping_an_existing_collection_uses_the_alias(self) -> None:
        config = milvus_config(alias="asclepius")
        config.collection.drop_existing = True
        manager = MilvusCollectionManager(config)

        with (
            patch(f"{MODULE}.utility") as utility,
            patch(f"{MODULE}.Collection"),
        ):
            utility.has_collection.return_value = True
            manager.create_collection()

        self.assertEqual(utility.drop_collection.call_args.kwargs["using"], "asclepius")

    def test_reusing_an_existing_collection_uses_the_alias(self) -> None:
        manager = MilvusCollectionManager(milvus_config(alias="asclepius"))

        with (
            patch(f"{MODULE}.utility") as utility,
            patch(f"{MODULE}.Collection") as collection,
        ):
            utility.has_collection.return_value = True
            collection.return_value.schema.fields = []
            manager.create_collection()

        self.assertEqual(collection.call_args.kwargs["using"], "asclepius")

    def test_connect_passes_the_configured_timeout(self) -> None:
        manager = MilvusCollectionManager(milvus_config())

        with patch(f"{MODULE}.connections") as connections:
            manager.connect()

        self.assertEqual(connections.connect.call_args.kwargs["timeout"], 30)


class IngestionRunTests(unittest.TestCase):
    def ingestion_pipeline(self):
        config = IngestionConfig(
            openai=OpenAIConfig(api_key="test-key", dimensions=DIMENSION),
            milvus=milvus_config(),
            search_text=SearchTextConfig(),
        )
        with patch(f"{MODULE}.OpenAI"):
            return VectorDBIngestionPipeline(config)

    def test_an_empty_record_list_raises_before_connecting(self) -> None:
        pipeline = self.ingestion_pipeline()

        with patch.object(pipeline.collection_manager, "connect") as connect:
            with self.assertRaises(ValueError) as caught:
                pipeline.run([])

        self.assertIn("No records to ingest", str(caught.exception))
        connect.assert_not_called()

    def test_a_dimension_mismatch_between_openai_and_milvus_is_rejected(self) -> None:
        config = IngestionConfig(
            openai=OpenAIConfig(api_key="test-key", dimensions=DIMENSION + 1),
            milvus=milvus_config(),
        )

        with patch(f"{MODULE}.OpenAI"):
            with self.assertRaises(ValueError) as caught:
                VectorDBIngestionPipeline(config)

        self.assertIn("Dimension mismatch", str(caught.exception))

    def test_the_embedded_text_is_the_text_that_gets_stored(self) -> None:
        """The defect this change exists to fix: one render, used for both."""
        pipeline = self.ingestion_pipeline()
        drugs = records()
        expected = [drug.generate_search_text(pipeline.config.search_text) for drug in drugs]
        pipeline.embedding_generator = MagicMock()
        pipeline.embedding_generator.generate_embeddings_batch.return_value = embeddings()
        pipeline.collection_manager = MagicMock()
        pipeline.collection_manager.insert_records.return_value = len(drugs)
        pipeline.collection_manager.get_collection_stats.return_value = {}

        pipeline.run(drugs)

        embedded = pipeline.embedding_generator.generate_embeddings_batch.call_args.args[0]
        stored = pipeline.collection_manager.insert_records.call_args.args[2]
        self.assertEqual(embedded, expected)
        self.assertEqual(stored, expected)
        self.assertEqual(embedded, stored)


if __name__ == "__main__":
    unittest.main()
