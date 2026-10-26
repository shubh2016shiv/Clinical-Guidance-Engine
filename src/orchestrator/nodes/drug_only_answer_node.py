"""Node (Path C): answer from the drug catalog alone, without asking the guidelines."""

from typing import Any

from langgraph.config import get_stream_writer

from src.drugs_agent import format_drug_answer_as_markdown_list

from ..schema import OrchestratorState


def make_drug_only_answer_node(drug_agent: Any):
    """Build the node. The graph calls the returned function with its shared state."""

    async def drug_only_answer(state: OrchestratorState) -> dict[str, Any]:
        """Search the catalog for the question itself and send back the drugs found.

        The guidelines are not asked at all, so this answer carries no citations.
        The drug agent does its own classification and name search, which is why
        the question goes to it unchanged.
        """
        send_text_to_user = get_stream_writer()

        drug_answer = await drug_agent.answer_drug_search_query(state["clinical_question"])

        # The list is built locally rather than written by a model, so there is
        # nothing to stream piece by piece; it is sent as one update.
        send_text_to_user(format_drug_answer_as_markdown_list(drug_answer))
        return {"drug_answer": drug_answer}

    return drug_only_answer
