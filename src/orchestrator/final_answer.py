"""Combine guideline citations and the drug agent's streamed answer."""

from src.clinical_guideline_agent.citations.inline_citations import (
    insert_file_citation_markers,
)
from src.clinical_guideline_agent.schema import FileCitation, GuidelineAnswer

from .schema import QuestionRoute


# ---------------------------------------------------------------------------
# NODE 04 DETAIL — Add guideline citations and join completed sections
# ---------------------------------------------------------------------------
# Runs after streaming to add guideline citation markers; stock text is already final.
# Catalog-only answers have no guideline sources. The final node updates history afterward.
def build_final_answer(
    *,
    route: QuestionRoute,
    guideline_answer: GuidelineAnswer | None,
    catalog_section_text: str,
) -> tuple[str, list[FileCitation]]:
    """Return combined answer text and numbered sources without updating history.

    Clinical routes require a completed guideline answer. Catalog-only answers need
    no guideline answer and have no sources. Drug text is kept exactly as streamed.
    """

    if route == "drug":
        # The guidelines were never asked, so there is nothing to cite and the stock
        # list is the whole answer.
        return catalog_section_text, []

    # ---------------------------------------------------------------------------
    # NODE 04A — Add guideline citations on clinical routes only
    # ---------------------------------------------------------------------------
    # Clinical paths require a completed guideline_answer. The graph order supplies
    # that precondition; this function does not invent a fallback.
    cited = insert_file_citation_markers(guideline_answer.text, guideline_answer.citations)
    sources = cited.sources
    final_text = cited.text
    if catalog_section_text:
        final_text += "\n\n" + catalog_section_text

    return final_text, sources
