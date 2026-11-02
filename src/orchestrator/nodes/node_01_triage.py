"""Node 01: triage. See ../README.md for both paths."""

from typing import Any

from langgraph.runtime import Runtime

from src.memory import select_recent_previous_messages_for_chat_model

from ..schema import OrchestratorContext, OrchestratorState
from ..triage import classify_question


# ---------------------------------------------------------------------------
# NODE 01 — Choose guidelines, drug catalog or both
# ---------------------------------------------------------------------------
# Read the question and stored history. Return initial route and product search terms.
# state is saved working data; runtime.context supplies the agents and model client.
async def triage(state: OrchestratorState, runtime: Runtime[OrchestratorContext]) -> dict[str, Any]:
    """Select recent history, classify the question and set the initial graph route."""
    question_triage = await classify_question(
        question=state["clinical_question"],
        previous_messages=select_recent_previous_messages_for_chat_model(
            state["conversation_messages"]
        ),
        chat_completions_provider=runtime.context.chat_completions_provider,
        model_name=runtime.context.model_name,
    )

    if question_triage.kind == "clinical":
        return {"route": "clinical", "drug_search_terms": []}

    # Stock queries come from the question/history, never from class extraction.
    search_terms = [term.strip() for term in question_triage.drug_search_terms if term.strip()]
    return {
        "route": "drug" if question_triage.kind == "catalog_only" else "clinical_and_drug",
        "drug_search_terms": search_terms or [state["clinical_question"]],
    }
