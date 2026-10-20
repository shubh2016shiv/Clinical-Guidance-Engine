"""Find drugs for a query, then have the model structure what was found."""

import json
from typing import Any

from src.config import get_settings
from src.prompts.drug_system_prompt import DRUG_SYSTEM_PROMPT

from .milvus_drug_search import DrugSearcher
from .schema import DrugAnswer, DrugMatch

# Low sampling values encourage the model to copy retrieved records.
DRUG_ANSWER_TEMPERATURE = 0.01
DRUG_ANSWER_TOP_P = 0.99


def _remove_empty_fields_from_drug_record(drug_record: dict[str, Any]) -> dict[str, Any]:
    """Drop the empty columns, so the model is never asked to copy a blank as null."""
    return {
        field_name: field_value
        for field_name, field_value in drug_record.items()
        if field_value not in (None, "", "[]")
    }


class DrugAgent:
    """Search the drug collection once, then return the rows as a ``DrugAnswer``."""

    def __init__(
        self,
        drug_searcher: DrugSearcher,
        chat_completions_provider: Any,
        *,
        chat_completion_model_name: str | None = None,
        drug_answer_system_prompt: str = DRUG_SYSTEM_PROMPT,
        chat_completion_temperature: float = DRUG_ANSWER_TEMPERATURE,
        chat_completion_top_p: float = DRUG_ANSWER_TOP_P,
    ) -> None:
        self.drug_searcher = drug_searcher
        self.chat_completions_provider = chat_completions_provider
        self.chat_completion_model_name = (
            chat_completion_model_name or get_settings().openai_model_name
        )
        self.drug_answer_system_prompt = drug_answer_system_prompt
        self.chat_completion_temperature = chat_completion_temperature
        self.chat_completion_top_p = chat_completion_top_p

    async def answer_drug_search_query(
        self, drug_search_query: str, *, max_retrieved_drug_records: int | None = None
    ) -> DrugAnswer:
        """Return drug matches backed by records found for the search query."""
        drug_search_results = await self.drug_searcher.search_drug_records(
            drug_name=drug_search_query, max_drug_results=max_retrieved_drug_records
        )
        retrieved_drug_records = [
            _remove_empty_fields_from_drug_record(search_result.fields)
            for search_result in drug_search_results
        ]
        chat_completion_result = await self.chat_completions_provider.execute_chat_completion(
            [
                {"role": "system", "content": self.drug_answer_system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"query": drug_search_query, "rows": retrieved_drug_records}
                    ),
                },
            ],
            model=self.chat_completion_model_name,
            temperature=self.chat_completion_temperature,
            top_p=self.chat_completion_top_p,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "drug_answer",
                    "schema": DrugAnswer.model_json_schema(),
                    "strict": True,
                },
            },
        )
        drug_answer = self._parse_chat_completion_as_drug_answer(chat_completion_result)
        self._validate_drug_matches_against_retrieved_records(drug_answer, retrieved_drug_records)
        return self._deduplicate_drug_matches(drug_answer)

    @staticmethod
    def _parse_chat_completion_as_drug_answer(chat_completion_result: Any) -> DrugAnswer:
        if chat_completion_result.finish_reason != "stop":
            raise RuntimeError(f"Drug response ended early: {chat_completion_result.finish_reason}")
        if not chat_completion_result.content:
            raise RuntimeError("Drug response had no text; the model may have refused")
        return DrugAnswer.model_validate_json(chat_completion_result.content)

    @staticmethod
    def _deduplicate_drug_matches(drug_answer: DrugAnswer) -> DrugAnswer:
        """Keep the first of any drug the model listed twice; each retrieved row is one drug."""
        unique_matches_by_name: dict[str, DrugMatch] = {}
        for drug_match in drug_answer.matches:
            unique_matches_by_name.setdefault(drug_match.drug_name, drug_match)
        return drug_answer.model_copy(update={"matches": list(unique_matches_by_name.values())})

    @staticmethod
    def _validate_drug_matches_against_retrieved_records(
        drug_answer: DrugAnswer, retrieved_drug_records: list[dict[str, Any]]
    ) -> None:
        """Reject any match whose drug name was not among the retrieved rows."""
        retrieved_drug_names = {record["drug_name"] for record in retrieved_drug_records}
        unretrieved_drug_names = [
            drug_match.drug_name
            for drug_match in drug_answer.matches
            if drug_match.drug_name not in retrieved_drug_names
        ]
        if unretrieved_drug_names:
            raise RuntimeError(
                f"Drug answer contains drugs that were not retrieved: {unretrieved_drug_names}"
            )
