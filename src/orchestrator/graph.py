"""Answers medical questions using clinical guidelines and, when needed, drug data.

THE BIG PICTURE
---------------
A user asks a question. We first decide: does this question ONLY need guideline
information, or does it ALSO need information about actual drugs?

Path A - guideline only
    1. Ask the guideline agent and stream its answer to the user as it is written.

Path B - guideline + drugs
    1. Ask the guideline agent for the complete answer (no streaming yet).
    2. Read that answer and pull out the drug classes it recommends
       (e.g. "ACE inhibitors", "thiazide diuretics").
    3. Search the drug catalog for each of those classes (all at once).
    4. Ask the model to write ONE final answer combining the guideline text
       and the drugs found, and stream it.
    5. Add a table of the drugs at the end.

The steps are wired together using LangGraph (a "graph" is just a flowchart
where each box is a function and arrows say which box runs next).
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from src.clinical_guideline_agent.citations.inline_citations import (
    drop_unknown_citation_markers,
    mark_citations_in_text,
)
from src.config import get_settings
from src.drugs_agent import DrugAnswer, DrugMatch, format_drug_answer_as_markdown_table
from src.prompts.orchestrator_system_prompts import (
    ANSWER_CONSOLIDATION_SYSTEM_PROMPT,
    DRUG_CLASS_EXTRACTION_SYSTEM_PROMPT,
    QUESTION_ROUTING_SYSTEM_PROMPT,
)

from .schema import (
    DrugLookupRoutingDecision,
    OrchestratorAnswer,
    OrchestratorState,
    OrchestratorUpdate,
    RecommendedDrugClasses,
)

# Guidelines often recommend several classes (hypertension can name four), so allow 5.
MAX_DRUG_CLASSES_TO_SEARCH = 5

# Very low temperature = the model gives almost the same answer every time.
TEMPERATURE = 0.01


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
        self.guideline_agent = guideline_agent
        self.drug_agent = drug_agent
        self.chat_completions_provider = chat_completions_provider
        # Use the model name we were given, otherwise fall back to the one in settings.
        self.model_name = chat_completion_model_name or get_settings().openai_model_name
        # Build the flowchart once, when the object is created.
        self.graph = self._build_graph()

    # =====================================================================
    # PUBLIC METHOD - this is what the rest of the app calls
    # =====================================================================

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

        # Step 3: read the final results out of the graph's data.
        guideline_answer = final_state["guideline_answer"]
        used_drug_path = final_state["needs_drug_information"]

        # Step 4: work out the final text and the list of sources (citations).
        if used_drug_path:
            # The model wrote this answer while looking at numbered citation markers.
            # Models sometimes invent a marker number, so remove any that don't match
            # a real source.
            sources = final_state["guideline_sources"]
            final_text = drop_unknown_citation_markers("".join(streamed_text_pieces), len(sources))
        else:
            # On the guideline-only path, the streamed text has no citation markers,
            # because the search tool only tells us where it cited things AFTER the
            # whole answer is finished. So we add the markers now.
            cited = mark_citations_in_text(guideline_answer.text, guideline_answer.citations)
            sources = cited.sources
            final_text = cited.text

        # Step 5: send the final "completed" update with everything bundled together.
        yield OrchestratorUpdate(
            type="completed",
            answer=OrchestratorAnswer(
                route="clinical_and_drug" if used_drug_path else "clinical",
                text=final_text,
                sources=sources,
                guideline_answer=guideline_answer,
                drug_classes=final_state.get("drug_classes", []),
                drug_answer=final_state.get("drug_answer"),
            ),
        )

    # =====================================================================
    # THE FLOWCHART - which function runs, and in what order
    # =====================================================================

    def _build_graph(self) -> Any:
        """Build the flowchart below. (*) marks a step that streams text to the user.

                                START
                                  |
                                  v
                         +-----------------+
                         |  decide_route   |   needs drug information?
                         +-----------------+
                             |         |
                 no (clinical)         (drug) yes
                             |         |
                             v         v
        +----------------------+   +--------------------------+
        | guideline_only_answer|   | get_full_guideline_answer|
        |     (*) streams      |   |   whole answer, + [1][2] |
        +----------------------+   +--------------------------+
                             |         |
                             |         v
                             |     +----------------------+
                             |     | extract_drug_classes |
                             |     +----------------------+
                             |         |
                             |         v
                             |     +----------------------+
                             |     |     search_drugs     |
                             |     |  all classes at once |
                             |     +----------------------+
                             |         |
                             |         v
                             |     +----------------------+
                             |     | write_combined_answer|
                             |     | (*) streams + table  |
                             |     +----------------------+
                             |         |
                             v         v
                                  END
        """
        graph = StateGraph(OrchestratorState)

        # Register every box (node) in the flowchart.
        graph.add_node("decide_route", self._decide_route)
        graph.add_node("guideline_only_answer", self._guideline_only_answer)
        graph.add_node("get_full_guideline_answer", self._get_full_guideline_answer)
        graph.add_node("extract_drug_classes", self._extract_drug_classes)
        graph.add_node("search_drugs", self._search_drugs)
        graph.add_node("write_combined_answer", self._write_combined_answer)

        # Draw the arrows.
        graph.add_edge(START, "decide_route")

        # A fork: after deciding, go to the drug path or the guideline-only path.
        graph.add_conditional_edges(
            "decide_route",
            lambda state: "drug" if state["needs_drug_information"] else "clinical",
            {
                "clinical": "guideline_only_answer",
                "drug": "get_full_guideline_answer",
            },
        )

        # Path A: one step, then done.
        graph.add_edge("guideline_only_answer", END)

        # Path B: a straight line of steps.
        graph.add_edge("get_full_guideline_answer", "extract_drug_classes")
        graph.add_edge("extract_drug_classes", "search_drugs")
        graph.add_edge("search_drugs", "write_combined_answer")
        graph.add_edge("write_combined_answer", END)

        return graph.compile()

    # =====================================================================
    # THE STEPS (each one is a box in the flowchart)
    # Each step receives the shared data ("state") and returns only the new
    # items it wants to add to it.
    # =====================================================================

    async def _decide_route(self, state: OrchestratorState) -> dict[str, Any]:
        """Ask the model: does this question also need drug information?"""
        decision = await self._ask_model_for_structured_json(
            QUESTION_ROUTING_SYSTEM_PROMPT,
            state["clinical_question"],
            DrugLookupRoutingDecision,
        )
        return {"needs_drug_information": decision.needs_drug_information}

    # ---------- Path A ----------

    async def _guideline_only_answer(self, state: OrchestratorState) -> dict[str, Any]:
        """Stream the guideline agent's answer straight to the user."""
        send_text_to_user = get_stream_writer()

        async for update in self.guideline_agent.stream_guideline_answer_updates(
            state["clinical_question"]
        ):
            if update.type == "text":
                send_text_to_user(update.text)  # forward each piece immediately
            elif update.answer is not None:
                # The stream has finished; save the complete answer.
                return {"guideline_answer": update.answer}

        # If we get here the stream ended without ever giving a final answer.
        raise RuntimeError("Guideline stream ended without a completed answer")

    # ---------- Path B ----------

    async def _get_full_guideline_answer(self, state: OrchestratorState) -> dict[str, Any]:
        """Get the whole guideline answer at once (not streamed), and add citation markers."""
        guideline_answer = await self.guideline_agent.answer_guideline_question(
            state["clinical_question"]
        )
        cited = mark_citations_in_text(guideline_answer.text, guideline_answer.citations)
        return {
            "guideline_answer": guideline_answer,
            "cited_guideline_text": cited.text,  # text with [1], [2]... markers
            "guideline_sources": cited.sources,  # the list those markers point to
        }

    async def _extract_drug_classes(self, state: OrchestratorState) -> dict[str, Any]:
        """Ask the model which drug classes the guideline answer recommends."""
        # Give the model both the question and the guideline's answer.
        model_input = json.dumps(
            {
                "question": state["clinical_question"],
                "guideline_answer": state["guideline_answer"].text,
            }
        )
        result = await self._ask_model_for_structured_json(
            DRUG_CLASS_EXTRACTION_SYSTEM_PROMPT,
            model_input,
            RecommendedDrugClasses,
        )

        # Clean up: trim spaces and throw away blank names.
        drug_classes = [name.strip() for name in result.drug_classes if name.strip()]

        # Keep only the first few so we don't search too many.
        return {"drug_classes": drug_classes[:MAX_DRUG_CLASSES_TO_SEARCH]}

    async def _search_drugs(self, state: OrchestratorState) -> dict[str, Any]:
        """Search the drug catalog for every class, all at the same time."""
        # asyncio.gather runs all the searches in parallel and waits for all of them.
        results_per_class = await asyncio.gather(
            *(self.drug_agent.answer_drug_class_query(name) for name in state["drug_classes"])
        )

        # The same drug can show up under several classes. Keep only the first copy
        # of each drug name.
        drugs_by_name: dict[str, DrugMatch] = {}
        for result in results_per_class:
            for drug in result.matches:
                drugs_by_name.setdefault(drug.drug_name, drug)

        all_drugs = list(drugs_by_name.values())
        return {"drug_answer": DrugAnswer(found=bool(all_drugs), matches=all_drugs)}

    async def _write_combined_answer(self, state: OrchestratorState) -> dict[str, Any]:
        """Have the model write one final answer, stream it, then append the drug table."""
        send_text_to_user = get_stream_writer()
        drug_answer = state["drug_answer"]

        # Everything the model needs to write the final answer.
        model_input = json.dumps(
            {
                "question": state["clinical_question"],
                "guideline_answer": state["cited_guideline_text"],
                "searched_drug_classes": state["drug_classes"],
                "drug_records": [
                    drug.model_dump(exclude_none=True) for drug in drug_answer.matches
                ],
            }
        )

        text_stream = await self.chat_completions_provider.execute_chat_completion(
            [
                {"role": "system", "content": ANSWER_CONSOLIDATION_SYSTEM_PROMPT},
                {"role": "user", "content": model_input},
            ],
            stream=True,
            model=self.model_name,
            temperature=TEMPERATURE,
        )

        # Forward the model's text to the user piece by piece.
        async for text_piece in text_stream:
            send_text_to_user(text_piece)

        # Finally, add the table of drugs under the answer.
        send_text_to_user("\n\n" + format_drug_answer_as_markdown_table(drug_answer))

        return {}  # nothing new to save; the text was already streamed out

    # =====================================================================
    # HELPER
    # =====================================================================

    async def _ask_model_for_structured_json(
        self, system_prompt: str, user_content: str, output_model: type[Any]
    ) -> Any:
        """Ask the model a question and force its reply into a fixed JSON shape.

        `output_model` describes the shape we want. We send that shape to the model,
        then parse the reply into that same class.
        """
        result = await self.chat_completions_provider.execute_chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            model=self.model_name,
            temperature=TEMPERATURE,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": output_model.__name__,
                    "schema": output_model.model_json_schema(),
                    "strict": True,
                },
            },
        )

        # Safety checks: the model must have finished normally and actually replied.
        if result.finish_reason != "stop":
            raise RuntimeError(f"{output_model.__name__} ended early: {result.finish_reason}")
        if not result.content:
            raise RuntimeError(f"{output_model.__name__} had no text; the model may have refused")

        return output_model.model_validate_json(result.content)
