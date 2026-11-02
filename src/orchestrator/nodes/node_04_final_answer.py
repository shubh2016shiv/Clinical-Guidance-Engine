"""Node 04: final answer. See ../README.md for both paths."""

from typing import Any

from langchain_core.messages import AIMessage

from ..final_answer import build_final_answer
from ..schema import OrchestratorState


# ---------------------------------------------------------------------------
# NODE 04 — Compose and remember the completed text
# ---------------------------------------------------------------------------
# Guideline citation markers are added; streamed drug text is unchanged. Return an AIMessage so the
# message reducer merges the finalized answer into the saved conversation.
async def finalize_answer(state: OrchestratorState) -> dict[str, Any]:
    """Assemble the answer and append its final text to graph conversation history."""
    final_text, sources = build_final_answer(
        route=state["route"],
        guideline_answer=state["guideline_answer"],
        catalog_section_text=state["catalog_section_text"],
    )

    return {
        "final_text": final_text,
        "sources": sources,
        # The `add_messages` reducer appends this right after the question, which
        # keeps the history in question, answer, question, answer order.
        "conversation_messages": [AIMessage(content=final_text)],
    }
