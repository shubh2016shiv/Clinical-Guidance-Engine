"""Retrieve drug records and stream an answer."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from src.config import get_settings
from src.logs import get_component_logger
from src.prompts.drug_system_prompt import DRUG_STREAM_SYSTEM_PROMPT

from .drug_classification import DrugClassifier
from .milvus_drug_search import DrugSearcher
from .retrieved_drug_formatter import format_retrieved_drug_table

# Low sampling values encourage the model to copy retrieved records.
DRUG_ANSWER_TEMPERATURE = 0.01
DRUG_ANSWER_TOP_P = 0.99
logger = get_component_logger("DrugAgent")


def _remove_empty_fields_from_drug_record(drug_record: dict[str, Any]) -> dict[str, Any]:
    """Drop the empty columns, so the model is never asked to copy a blank as null."""
    return {
        field_name: field_value
        for field_name, field_value in drug_record.items()
        if field_value not in (None, "", "[]")
    }


class DrugAgent:
    """Own drug retrieval, interpretation and user-facing answer generation."""

    def __init__(
        self,
        drug_searcher: DrugSearcher,
        chat_completions_provider: Any,
        *,
        chat_completion_model_name: str | None = None,
        chat_completion_temperature: float = DRUG_ANSWER_TEMPERATURE,
        chat_completion_top_p: float = DRUG_ANSWER_TOP_P,
        drug_classifier: DrugClassifier | None = None,
    ) -> None:
        self.drug_searcher = drug_searcher
        self.chat_completions_provider = chat_completions_provider
        self.chat_completion_model_name = (
            chat_completion_model_name or get_settings().openai_model_name
        )
        self.chat_completion_temperature = chat_completion_temperature
        self.chat_completion_top_p = chat_completion_top_p
        self.drug_classifier = drug_classifier or DrugClassifier(
            chat_completions_provider,
            chat_completion_model_name=self.chat_completion_model_name,
        )

    async def _retrieve_drug_records(
        self, drug_search_query: str, *, max_retrieved_drug_records: int | None = None
    ) -> list[Any]:
        """Classify the query and retrieve matching catalog candidates."""
        identified_drug_class = await self.drug_classifier.identify_drug_class(drug_search_query)
        logger.info("Drug classification completed", drug_class=identified_drug_class)
        return await self.drug_searcher.search_drug_records(
            drug_name=drug_search_query,
            drug_class=identified_drug_class,
            max_drug_results=max_retrieved_drug_records,
        )

    # ---------------------------------------------------------------------------
    # STREAM AN ANSWER — Retrieve first; the table is context, not display output
    # ---------------------------------------------------------------------------
    async def stream_drug_answer_text(
        self, *, question: str, drug_search_terms: list[str]
    ) -> AsyncIterator[str]:
        """Send retrieved rows to one answer call and forward its text as it arrives.

        Search terms may resolve references from conversation history. The original
        question tells the model what to explain about those records. This path does
        not generate JSON for later rendering.
        """
        search_results_by_term = await asyncio.gather(
            *(self._retrieve_drug_records(term) for term in drug_search_terms)
        )
        retrieved_records: list[dict[str, Any]] = []
        for search_results in search_results_by_term:
            for search_result in search_results:
                record = _remove_empty_fields_from_drug_record(search_result.fields)
                # Drop identical rows across queries, preserving distinct forms/strengths.
                if record not in retrieved_records:
                    retrieved_records.append(record)

        answer_stream = await self.chat_completions_provider.execute_chat_completion(
            [
                {"role": "system", "content": DRUG_STREAM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {json.dumps(question, ensure_ascii=False)}\n"
                        f"Resolved search queries: {json.dumps(drug_search_terms, ensure_ascii=False)}"
                        "\n\nRetrieved drug records:\n"
                        f"{format_retrieved_drug_table(retrieved_records)}"
                    ),
                },
            ],
            model=self.chat_completion_model_name,
            temperature=self.chat_completion_temperature,
            top_p=self.chat_completion_top_p,
            stream=True,
        )
        received_answer_text = False
        async for text_chunk in answer_stream:
            if text_chunk:
                received_answer_text = received_answer_text or bool(text_chunk.strip())
                yield text_chunk
        if not received_answer_text:
            raise RuntimeError("Drug answer stream ended without any answer text")
