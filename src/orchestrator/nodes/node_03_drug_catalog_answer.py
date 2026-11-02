"""Node 03: drug catalog answer. See ../README.md for both paths."""

from typing import Any

from langgraph.runtime import Runtime

from ..schema import OrchestratorContext, OrchestratorState
from ..stream_updates import send_answer_text, send_status


# ---------------------------------------------------------------------------
# NODE 03 — Look up drugs by class or product search term
# ---------------------------------------------------------------------------
# Both stock paths meet here. The drug agent owns retrieval and answer generation.
async def search_catalog(
    state: OrchestratorState, runtime: Runtime[OrchestratorContext]
) -> dict[str, Any]:
    """Forward the drug agent's live answer and save the same text for finalization."""
    drug_search_terms = state["drug_search_terms"]
    send_status(f"Searching the catalog for {', '.join(drug_search_terms)}...")
    separator = "" if state["route"] == "drug" else "\n\n"
    answer_chunks = []
    async for text_chunk in runtime.context.drug_agent.stream_drug_answer_text(
        question=state["clinical_question"], drug_search_terms=drug_search_terms
    ):
        if not text_chunk:
            continue
        # Add the separator only with the first real chunk, never as an empty section.
        send_answer_text(separator + text_chunk)
        separator = ""
        answer_chunks.append(text_chunk)
    stock_text = "".join(answer_chunks)
    if not stock_text.strip():
        raise RuntimeError("Drug answer stream ended without any answer text")
    return {"catalog_section_text": stock_text}
