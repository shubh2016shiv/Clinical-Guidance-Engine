"""Node (Path B, step 2): find the drug classes the guideline answer recommends."""

import json
from typing import Any

from src.prompts.orchestrator_system_prompts import DRUG_CLASS_EXTRACTION_SYSTEM_PROMPT

from ..schema import OrchestratorState, RecommendedDrugClasses
from ..structured_model_call import ask_model_for_structured_json

# Guidelines often recommend several classes (hypertension can name four), so allow 5.
MAX_DRUG_CLASSES_TO_SEARCH = 5


def make_extract_drug_classes_node(chat_completions_provider: Any, model_name: str):
    """Build the node. The graph calls the returned function with its shared state."""

    async def extract_drug_classes(state: OrchestratorState) -> dict[str, Any]:
        """Ask the model which drug classes the guideline answer recommends."""
        # Give the model both the question and the guideline's answer.
        model_input = json.dumps(
            {
                "question": state["clinical_question"],
                "guideline_answer": state["guideline_answer"].text,
            }
        )
        result = await ask_model_for_structured_json(
            chat_completions_provider,
            model_name,
            DRUG_CLASS_EXTRACTION_SYSTEM_PROMPT,
            model_input,
            RecommendedDrugClasses,
        )

        # Clean up: trim spaces and throw away blank names.
        drug_classes = [name.strip() for name in result.drug_classes if name.strip()]

        # Keep only the first few so we don't search too many.
        return {"drug_classes": drug_classes[:MAX_DRUG_CLASSES_TO_SEARCH]}

    return extract_drug_classes
