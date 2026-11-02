"""Node 02: guideline answer. See ../README.md for both paths."""

from typing import Any

from langgraph.runtime import Runtime

from ..schema import OrchestratorContext, OrchestratorState
from ..stream_updates import send_answer_text, send_status


# ---------------------------------------------------------------------------
# NODE 02 — Stream guidelines on clinical and combined routes
# ---------------------------------------------------------------------------
# Forward text immediately; store the completed answer for final answer assembly and citations.
async def stream_guideline_answer(
    state: OrchestratorState, runtime: Runtime[OrchestratorContext]
) -> dict[str, Any]:
    """Call the guideline agent, forward text immediately and store its completed answer.

    Retrieval and clinical interpretation stay inside the agent. This node only
    coordinates its updates. Missing completion raises after forwarding partial text.
    """
    # NODE 02A — Announce the work before waiting for the guideline agent.
    send_status("Searching the clinical guidelines...")

    # NODE 02B — Forward each piece before requesting the agent's next update.
    async for update in runtime.context.guideline_agent.stream_guideline_answer_updates(
        state["clinical_question"]
    ):
        if update.type == "text":
            send_answer_text(update.text)
        elif update.answer is not None:
            # NODE 02C — Return the full answer for final answer assembly and citations.
            return {"guideline_answer": update.answer}

    raise RuntimeError("Guideline stream ended without a completed answer")
