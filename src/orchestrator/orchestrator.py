"""Answers medical questions using clinical guidelines and, when needed, drug data.

A user asks a question. The flowchart in `graph.py` decides which agents it needs
and runs them. This class runs that flowchart, passes the streamed text on to the
caller, and bundles the final answer together with its sources.
"""

from collections.abc import AsyncIterator
from typing import Any

from src.clinical_guideline_agent.citations.inline_citations import (
    insert_file_citation_markers,
    remove_unknown_file_citation_markers,
)
from src.config import get_settings

from .graph import build_orchestrator_graph
from .schema import OrchestratorAnswer, OrchestratorState, OrchestratorUpdate


class ClinicalDrugOrchestrator:
    """Sends a question through the guideline agent and, if needed, the drug agent."""

    def __init__(
        self,
        guideline_agent: Any,
        drug_agent: Any,
        chat_completions_provider: Any,
        *,
        chat_completion_model_name: str | None = None,
    ) -> None:
        # Use the model name we were given, otherwise fall back to the one in settings.
        model_name = chat_completion_model_name or get_settings().openai_model_name

        # Build the flowchart once, when the object is created.
        self.graph = build_orchestrator_graph(
            guideline_agent, drug_agent, chat_completions_provider, model_name
        )

    async def stream_clinical_drug_answer_updates(
        self, clinical_question: str
    ) -> AsyncIterator[OrchestratorUpdate]:
        """Yield text pieces as they are written, then one final update with everything."""

        # Step 1: reject empty questions.
        if not clinical_question.strip():
            raise ValueError("A question is required")

        # Step 2: run the flowchart and forward everything it streams out.
        streamed_text_pieces: list[str] = []  # we keep these to rebuild the full answer later
        final_state: OrchestratorState = {}  # the graph's data once it has finished

        async for mode, payload in self.graph.astream(
            {"clinical_question": clinical_question},
            stream_mode=["custom", "values"],
        ):
            if mode == "custom":
                # "custom" = a piece of answer text. Pass it straight to the caller.
                streamed_text_pieces.append(payload)
                yield OrchestratorUpdate(type="text", text=payload)
            else:
                # "values" = the graph's current data. The last one we see is the final one.
                final_state = payload

        # Step 3: read the final results out of the graph's data. There is no
        # guideline answer on the drug-only route, which never asks the guidelines.
        route = final_state["route"]
        guideline_answer = final_state.get("guideline_answer")

        # Step 4: work out the final text and the list of sources (citations).
        if route == "drug":
            # Nothing was cited, so the streamed drug list is already the answer.
            final_text = "".join(streamed_text_pieces)
            sources = []
        elif route == "clinical_and_drug":
            # The model wrote this answer while looking at numbered citation markers.
            # Models sometimes invent a marker number, so remove any that don't match
            # a real source.
            sources = final_state["guideline_sources"]
            final_text = remove_unknown_file_citation_markers(
                "".join(streamed_text_pieces), len(sources)
            )
        else:
            # On the guideline-only path, the streamed text has no citation markers,
            # because the search tool only tells us where it cited things AFTER the
            # whole answer is finished. So we add the markers now.
            cited = insert_file_citation_markers(guideline_answer.text, guideline_answer.citations)
            sources = cited.sources
            final_text = cited.text

        # Step 5: send the final "completed" update with everything bundled together.
        yield OrchestratorUpdate(
            type="completed",
            answer=OrchestratorAnswer(
                route=route,
                text=final_text,
                sources=sources,
                guideline_answer=guideline_answer,
                drug_classes=final_state.get("drug_classes", []),
                drug_answer=final_state.get("drug_answer"),
            ),
        )
