"""Node (Path B, step 1): fetch the complete guideline answer, without streaming it.

The user does not see this text directly. The next steps read it to find the
recommended drug classes, and the final answer is written later from it plus the
drugs found. Compare `stream_guideline_answer_node`, which shows the answer to
the user as it is written.
"""

from typing import Any

from src.clinical_guideline_agent.citations.inline_citations import insert_file_citation_markers

from ..schema import OrchestratorState


def make_fetch_guideline_answer_node(guideline_agent: Any):
    """Build the node. The graph calls the returned function with its shared state."""

    async def fetch_guideline_answer(state: OrchestratorState) -> dict[str, Any]:
        """Get the whole guideline answer at once and add citation markers to it."""
        guideline_answer = await guideline_agent.answer_guideline_question(
            state["clinical_question"]
        )
        cited = insert_file_citation_markers(guideline_answer.text, guideline_answer.citations)
        return {
            "guideline_answer": guideline_answer,
            "cited_guideline_text": cited.text,  # text with [1], [2]... markers
            "guideline_sources": cited.sources,  # the list those markers point to
        }

    return fetch_guideline_answer
