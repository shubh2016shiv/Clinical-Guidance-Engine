import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pymilvus import FunctionType

from src.adapters.vector_store.exceptions import VectorDBCollectionError
from src.drugs_agent import (
    DrugAgent,
    DrugClassifier,
    DrugSearcher,
    verify_drug_collection_ready_for_search,
)
from src.drugs_agent.milvus_drug_search import build_drug_embedding_query_text
from src.drugs_agent.retrieved_drug_formatter import MAX_DRUG_ROWS
from src.prompts.drug_classification_system_prompt import DRUG_CLASSIFICATION_SYSTEM_PROMPT

OUTPUT_FIELDS = ["drug_name", "drug_class"]


def schema(*, bm25=True):
    functions = []
    if bm25:
        functions.append({"type": FunctionType.BM25, "output_field_names": ["drug_name_sparse"]})
    return {"functions": functions}


def provider(count=1, *, bm25=True):
    mock = MagicMock()
    mock.collection_name = "pharmaceutical_drugs"
    mock.connect = AsyncMock()
    mock.check_collection = AsyncMock()
    mock.count_entities = AsyncMock(return_value=count)
    mock.describe_collection = AsyncMock(return_value=schema(bm25=bm25))
    return mock


class VerifyDrugCollectionReadyForSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_the_row_count_when_data_is_present(self) -> None:
        self.assertEqual(await verify_drug_collection_ready_for_search(provider(6279)), 6279)

    async def test_an_empty_collection_raises_and_names_the_ingestion_script(self) -> None:
        with self.assertRaisesRegex(VectorDBCollectionError, "empty.*drug_ingestion_pipeline"):
            await verify_drug_collection_ready_for_search(provider(0))

    async def test_a_failed_collection_check_stops_before_counting(self) -> None:
        mock = provider(10)
        mock.check_collection.side_effect = VectorDBCollectionError("dimension mismatch")

        with self.assertRaises(VectorDBCollectionError):
            await verify_drug_collection_ready_for_search(mock)

        mock.count_entities.assert_not_awaited()

    async def test_a_collection_built_before_hybrid_search_asks_for_re_ingestion(self) -> None:
        mock = provider(6279, bm25=False)

        with self.assertRaisesRegex(VectorDBCollectionError, "no BM25 field.*Re-run"):
            await verify_drug_collection_ready_for_search(mock)

        mock.count_entities.assert_not_awaited()


class QueryTextTests(unittest.TestCase):
    def test_labels_match_the_ingested_search_text(self) -> None:
        self.assertEqual(
            build_drug_embedding_query_text("metformin", "antidiabetic"),
            "Drug: metformin | Class: antidiabetic",
        )

    def test_a_single_part_has_no_dangling_separator(self) -> None:
        self.assertEqual(build_drug_embedding_query_text("metformin", None), "Drug: metformin")
        self.assertEqual(
            build_drug_embedding_query_text(None, "antidiabetic"), "Class: antidiabetic"
        )

    def test_blank_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_drug_embedding_query_text("  ", None)
        with self.assertRaises(ValueError):
            build_drug_embedding_query_text(None, None)


def hit(drug_id, distance, name):
    return {
        "id": drug_id,
        "distance": distance,
        "entity": {"drug_name": name, "drug_class": "antidiabetic"},
    }


def searcher(raw_results=None):
    mock = MagicMock()
    mock.collection_name = "pharmaceutical_drugs"
    mock.config.search.top_k = 10
    mock.config.search.search_params = {"ef": 64}
    mock.config.search.output_fields = OUTPUT_FIELDS
    mock.embeddings.create_text_embedding = AsyncMock(return_value=[0.1, 0.2])
    mock.search_by_text = AsyncMock(return_value=["dense only"])
    client = MagicMock()
    client.hybrid_search.return_value = [raw_results or []]
    mock.get_client = AsyncMock(return_value=client)
    return DrugSearcher(mock, rrf_rank_constant=42), mock, client


class DrugSearcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_name_runs_one_hybrid_search_of_a_dense_and_a_bm25_request(self) -> None:
        drug_searcher, mock, collection = searcher([hit(7, 0.0328, "Metformin")])

        with patch("src.drugs_agent.milvus_drug_search.RRFRanker") as ranker:
            results = await drug_searcher.search_drug_records(
                drug_name=" metformin ", drug_class="antidiabetic"
            )

        mock.embeddings.create_text_embedding.assert_awaited_once_with(
            "Drug: metformin | Class: antidiabetic"
        )
        ranker.assert_called_once_with(42)
        _, kwargs = collection.hybrid_search.call_args
        requests, rerank = kwargs["reqs"], kwargs["ranker"]
        dense, sparse = requests
        self.assertEqual((dense.anns_field, dense.data), ("embedding", [[0.1, 0.2]]))
        self.assertEqual((sparse.anns_field, sparse.data), ("drug_name_sparse", ["metformin"]))
        self.assertEqual(sparse.param, {"metric_type": "BM25"})
        self.assertIs(rerank, ranker.return_value)
        self.assertEqual(kwargs["limit"], 10)
        self.assertEqual(kwargs["output_fields"], OUTPUT_FIELDS)
        self.assertEqual(results[0].id, 7)
        self.assertEqual(results[0].fields["drug_name"], "Metformin")

    async def test_each_ranking_fetches_more_candidates_than_the_requested_limit(self) -> None:
        drug_searcher, _, collection = searcher()

        await drug_searcher.search_drug_records(drug_name="metformin", max_drug_results=20)

        dense, sparse = collection.hybrid_search.call_args.kwargs["reqs"]
        self.assertEqual((dense.limit, sparse.limit), (60, 60))
        self.assertEqual(collection.hybrid_search.call_args.kwargs["limit"], 20)

    async def test_a_class_alone_is_a_plain_vector_search(self) -> None:
        drug_searcher, mock, collection = searcher()

        results = await drug_searcher.search_drug_records(
            drug_class="antidiabetic", max_drug_results=5
        )

        self.assertEqual(results, ["dense only"])
        mock.search_by_text.assert_awaited_once_with("Class: antidiabetic", limit=5)
        collection.hybrid_search.assert_not_called()

    async def test_nothing_to_search_for_is_rejected(self) -> None:
        drug_searcher, _, _ = searcher()

        with self.assertRaises(ValueError):
            await drug_searcher.search_drug_records()


class DrugClassifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_chat_completions_with_a_strict_classification_schema(self) -> None:
        chat_provider = MagicMock()
        chat_provider.execute_chat_completion = AsyncMock(
            return_value=SimpleNamespace(
                finish_reason="stop", content=json.dumps({"drug_class": "ACE inhibitors"})
            )
        )
        classifier = DrugClassifier(chat_provider, chat_completion_model_name="test-model")

        drug_class = await classifier.identify_drug_class("lisinopril")

        self.assertEqual(drug_class, "ACE inhibitors")
        messages = chat_provider.execute_chat_completion.call_args.args[0]
        self.assertEqual(messages[0]["content"], DRUG_CLASSIFICATION_SYSTEM_PROMPT)
        self.assertEqual(messages[1], {"role": "user", "content": "lisinopril"})
        options = chat_provider.execute_chat_completion.call_args.kwargs
        self.assertEqual(options["model"], "test-model")
        self.assertTrue(options["response_format"]["json_schema"]["strict"])
        self.assertEqual(
            set(options["response_format"]["json_schema"]["schema"]["properties"]),
            {"drug_class"},
        )

    async def test_unclassified_query_returns_none(self) -> None:
        chat_provider = MagicMock()
        chat_provider.execute_chat_completion = AsyncMock(
            return_value=SimpleNamespace(finish_reason="stop", content='{"drug_class": null}')
        )

        self.assertIsNone(
            await DrugClassifier(
                chat_provider, chat_completion_model_name="test-model"
            ).identify_drug_class("drugs for high blood pressure")
        )

    async def test_incomplete_classification_is_rejected(self) -> None:
        chat_provider = MagicMock()
        chat_provider.execute_chat_completion = AsyncMock(
            return_value=SimpleNamespace(finish_reason="length", content="{")
        )

        with self.assertRaisesRegex(RuntimeError, "ended early: length"):
            await DrugClassifier(
                chat_provider, chat_completion_model_name="test-model"
            ).identify_drug_class("lisinopril")


class DrugAgentRowCapTests(unittest.IsolatedAsyncioTestCase):
    async def _model_input_for(self, row_count: int) -> str:
        async def one_chunk():
            yield "ok"

        records = [SimpleNamespace(fields={"drug_name": f"drug {n}"}) for n in range(row_count)]
        chat_provider = MagicMock()
        chat_provider.execute_chat_completion = AsyncMock(return_value=one_chunk())
        drug_searcher = MagicMock()
        drug_searcher.search_drug_records = AsyncMock(return_value=records)
        agent = DrugAgent(
            drug_searcher,
            chat_provider,
            chat_completion_model_name="test-model",
            drug_classifier=MagicMock(identify_drug_class=AsyncMock(return_value=None)),
        )

        async for _ in agent.stream_drug_answer_text(question="q", drug_search_terms=["x"]):
            pass
        return chat_provider.execute_chat_completion.call_args.args[0][1]["content"]

    async def test_a_long_result_is_cut_to_the_cap_with_a_note(self) -> None:
        model_input = await self._model_input_for(MAX_DRUG_ROWS + 5)

        self.assertIn(f"drug {MAX_DRUG_ROWS - 1}", model_input)
        self.assertNotIn(f"drug {MAX_DRUG_ROWS} ", model_input + " ")
        self.assertIn(f"closest {MAX_DRUG_ROWS} of {MAX_DRUG_ROWS + 5}", model_input)

    async def test_a_short_result_is_sent_whole_without_a_note(self) -> None:
        model_input = await self._model_input_for(3)

        self.assertIn("drug 2", model_input)
        self.assertNotIn("NOTE", model_input)


if __name__ == "__main__":
    unittest.main()
