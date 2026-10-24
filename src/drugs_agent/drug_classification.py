"""Identify a drug class from a query with Chat Completions."""

from typing import Any

from src.config import get_settings
from src.prompts.drug_classification_system_prompt import DRUG_CLASSIFICATION_SYSTEM_PROMPT

from .schema import DrugClassification


class DrugClassifier:
    """Return an optional class to improve drug retrieval."""

    def __init__(
        self,
        chat_completions_provider: Any,
        *,
        chat_completion_model_name: str | None = None,
        drug_classification_system_prompt: str = DRUG_CLASSIFICATION_SYSTEM_PROMPT,
    ) -> None:
        self.chat_completions_provider = chat_completions_provider
        self.chat_completion_model_name = (
            chat_completion_model_name or get_settings().openai_model_name
        )
        self.drug_classification_system_prompt = drug_classification_system_prompt

    async def identify_drug_class(self, drug_search_query: str) -> str | None:
        """Classify a named product, or return None when no class is justified."""
        chat_completion_result = await self.chat_completions_provider.execute_chat_completion(
            [
                {"role": "system", "content": self.drug_classification_system_prompt},
                {"role": "user", "content": drug_search_query},
            ],
            model=self.chat_completion_model_name,
            temperature=0.01,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "drug_classification",
                    "schema": DrugClassification.model_json_schema(),
                    "strict": True,
                },
            },
        )
        if chat_completion_result.finish_reason != "stop":
            raise RuntimeError(
                f"Drug classification ended early: {chat_completion_result.finish_reason}"
            )
        if not chat_completion_result.content:
            raise RuntimeError("Drug classification had no text; the model may have refused")

        classification = DrugClassification.model_validate_json(chat_completion_result.content)
        if classification.drug_class is None:
            return None
        return classification.drug_class.strip() or None
