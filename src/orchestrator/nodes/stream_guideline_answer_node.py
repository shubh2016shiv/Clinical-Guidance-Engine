"""Node (Path A): stream the guideline agent's answer straight to the user.

This is the whole answer for guideline-only questions, so the text is sent to
the user while it is still being written. Compare `fetch_guideline_answer_node`,
which waits for the full answer because other steps still have to read it.
"""

from typing import Any

from langgraph.config import get_stream_writer

from ..schema import OrchestratorState


def make_stream_guideline_answer_node(guideline_agent: Any):
    """Build the node. The graph calls the returned function with its shared state."""

    async def stream_guideline_answer(state: OrchestratorState) -> dict[str, Any]:
        """Forward each piece of the guideline answer to the user as it is written."""
        send_text_to_user = get_stream_writer()

        async for update in guideline_agent.stream_guideline_answer_updates(
            state["clinical_question"]
        ):
            if update.type == "text":
                send_text_to_user(update.text)  # forward each piece immediately
            elif update.answer is not None:
                # The stream has finished; save the complete answer.
                return {"guideline_answer": update.answer}

        # If we get here the stream ended without ever giving a final answer.
        raise RuntimeError("Guideline stream ended without a completed answer")

    return stream_guideline_answer
