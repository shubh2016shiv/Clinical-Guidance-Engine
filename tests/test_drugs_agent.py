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
from src.prompts.drug_classification_system_prompt import DRUG_CLASSIFICATION_SYSTEM_PROMPT
from src.prompts.drug_system_prompt import DRUG_SYSTEM_PROMPT

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


ROW = {"drug_name": "Lisinopril", "drug_class": "ACE INHIBITORS, PLAIN", "formulation": "tablet"}


def match(name="Lisinopril"):
    return {
        "drug_name": name,
        "drug_class": "ACE INHIBITORS, PLAIN",
        "drug_sub_class": None,
        "therapeutic_category": None,
        "route_of_administration": "oral",
        "formulation": "tablet",
        "dosage_strengths": ["10 mg"],
    }


def agent_with(model_output, *, finish_reason="stop", rows=(ROW,)):
    searcher = MagicMock()
    searcher.search_drug_records = AsyncMock(
        return_value=[SimpleNamespace(fields=row) for row in rows]
    )
    response = SimpleNamespace(finish_reason=finish_reason, content=model_output)
    llm = MagicMock()
    llm.execute_chat_completion = AsyncMock(return_value=response)
    classifier = MagicMock()
    classifier.identify_drug_class = AsyncMock(return_value=None)
    return (
        DrugAgent(
            searcher,
            llm,
            chat_completion_model_name="test-model",
            drug_classifier=classifier,
        ),
        searcher,
        llm,
    )


class DrugAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_searches_once_then_asks_the_model_to_structure_the_rows(self) -> None:
        output = json.dumps({"found": True, "matches": [match()]})
        agent, searcher, llm = agent_with(output)

        answer = await agent.answer_drug_search_query("lisinopril", max_retrieved_drug_records=5)

        searcher.search_drug_records.assert_awaited_once_with(
            drug_name="lisinopril", drug_class=None, max_drug_results=5
        )
        agent.drug_classifier.identify_drug_class.assert_awaited_once_with("lisinopril")
        llm.execute_chat_completion.assert_awaited_once()
        (messages,), kwargs = llm.execute_chat_completion.call_args
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], DRUG_SYSTEM_PROMPT)
        self.assertEqual(json.loads(messages[1]["content"]), {"query": "lisinopril", "rows": [ROW]})
        self.assertEqual(kwargs["model"], "test-model")
        self.assertTrue(kwargs["response_format"]["json_schema"]["strict"])
        self.assertEqual(kwargs["temperature"], 0.01)
        self.assertEqual(kwargs["top_p"], 0.99)
        self.assertTrue(answer.found)
        self.assertEqual(answer.matches[0].drug_name, "Lisinopril")

    async def test_the_answer_has_no_free_text_field_the_model_could_make_claims_in(self) -> None:
        agent, _, llm = agent_with(json.dumps({"found": False, "matches": []}))

        await agent.answer_drug_search_query("zzz")

        response_format = llm.execute_chat_completion.call_args.kwargs["response_format"]
        schema = response_format["json_schema"]["schema"]
        self.assertEqual(set(schema["properties"]), {"found", "matches"})

    async def test_empty_columns_are_left_out_of_what_the_model_is_shown(self) -> None:
        sparse = {"drug_name": "Metformin", "formulation": "", "dosage_strengths": "[]"}
        agent, _, llm = agent_with(json.dumps({"found": False, "matches": []}), rows=(sparse,))

        await agent.answer_drug_search_query("metformin")

        sent = json.loads(llm.execute_chat_completion.call_args.args[0][1]["content"])
        self.assertEqual(sent["rows"], [{"drug_name": "Metformin"}])

    async def test_a_drug_the_model_lists_twice_is_returned_once_in_its_first_position(
        self,
    ) -> None:
        output = json.dumps(
            {
                "found": True,
                "matches": [match("Lisinopril"), match("Enalapril"), match("Lisinopril")],
            }
        )
        rows = (ROW, {"drug_name": "Enalapril"})
        agent, _, _ = agent_with(output, rows=rows)

        answer = await agent.answer_drug_search_query("ace inhibitors")

        self.assertEqual([m.drug_name for m in answer.matches], ["Lisinopril", "Enalapril"])

    async def test_a_class_query_searches_by_class_only_and_skips_the_classifier(self) -> None:
        agent, searcher, llm = agent_with(json.dumps({"found": True, "matches": [match()]}))
        agent.drug_classifier = MagicMock()
        agent.drug_classifier.identify_drug_class = AsyncMock()

        answer = await agent.answer_drug_class_query("ACE inhibitors")

        searcher.search_drug_records.assert_awaited_once_with(
            drug_class="ACE inhibitors", max_drug_results=None
        )
        agent.drug_classifier.identify_drug_class.assert_not_awaited()
        sent = json.loads(llm.execute_chat_completion.call_args.args[0][1]["content"])
        self.assertEqual(sent["query"], "ACE inhibitors")
        self.assertEqual(answer.matches[0].drug_name, "Lisinopril")

    async def test_a_drug_that_was_not_retrieved_is_rejected(self) -> None:
        output = json.dumps({"found": True, "matches": [match("Warfarin")]})
        agent, _, _ = agent_with(output)

        with self.assertRaisesRegex(RuntimeError, "not retrieved.*Warfarin"):
            await agent.answer_drug_search_query("lisinopril")

    async def test_a_response_cut_off_before_it_finished_is_an_error(self) -> None:
        agent, _, _ = agent_with("{", finish_reason="length")

        with self.assertRaisesRegex(RuntimeError, "ended early: length"):
            await agent.answer_drug_search_query("lisinopril")

    async def test_a_response_without_text_is_an_error(self) -> None:
        agent, _, _ = agent_with("")

        with self.assertRaisesRegex(RuntimeError, "no text"):
            await agent.answer_drug_search_query("lisinopril")

    async def test_identified_class_is_passed_to_drug_search(self) -> None:
        agent, searcher, _ = agent_with(json.dumps({"found": True, "matches": [match()]}))
        agent.drug_classifier.identify_drug_class.return_value = "ACE inhibitors"

        with patch("src.drugs_agent.agent.logger") as drug_agent_logger:
            await agent.answer_drug_search_query("lisinopril")

        searcher.search_drug_records.assert_awaited_once_with(
            drug_name="lisinopril", drug_class="ACE inhibitors", max_drug_results=None
        )
        drug_agent_logger.info.assert_called_once_with(
            "Drug classification completed", drug_class="ACE inhibitors"
        )

    async def test_default_classifier_runs_before_search_and_final_answer(self) -> None:
        drug_searcher = MagicMock()
        drug_searcher.search_drug_records = AsyncMock(return_value=[SimpleNamespace(fields=ROW)])
        chat_provider = MagicMock()
        chat_provider.execute_chat_completion = AsyncMock(
            side_effect=[
                SimpleNamespace(finish_reason="stop", content='{"drug_class": "ACE inhibitors"}'),
                SimpleNamespace(
                    finish_reason="stop",
                    content=json.dumps({"found": True, "matches": [match()]}),
                ),
            ]
        )
        agent = DrugAgent(drug_searcher, chat_provider, chat_completion_model_name="test-model")

        drug_answer = await agent.answer_drug_search_query("lisinopril")

        self.assertEqual(drug_answer.matches[0].drug_name, "Lisinopril")
        drug_searcher.search_drug_records.assert_awaited_once_with(
            drug_name="lisinopril", drug_class="ACE inhibitors", max_drug_results=None
        )
        self.assertEqual(chat_provider.execute_chat_completion.await_count, 2)
        first_messages = chat_provider.execute_chat_completion.await_args_list[0].args[0]
        self.assertEqual(first_messages[1]["content"], "lisinopril")


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


if __name__ == "__main__":
    unittest.main()
