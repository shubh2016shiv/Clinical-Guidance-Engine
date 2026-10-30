"""Answers medical questions using clinical guidelines and, when needed, drug data.

A user asks a question. The flowchart in `graph.py` decides which agents it needs
and runs them. This class runs that flowchart, passes the streamed text on to the
caller, and bundles the final answer together with its sources.
"""

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.clinical_guideline_agent.citations.inline_citations import (
    insert_file_citation_markers,
    remove_unknown_file_citation_markers,
)
from src.config import get_settings

from .graph import build_orchestrator_graph
from .schema import OrchestratorAnswer, OrchestratorState, OrchestratorUpdate

# The checkpointer saves the graph's data to storage and reads it back. For our own
# classes it only does so when they are listed here, as (module, class name) pairs.
# Without this list LangGraph logs a warning and will refuse to load them in future.
CHECKPOINT_SAVED_CLASSES = [
    ("src.clinical_guideline_agent.schema", "GuidelineAnswer"),
    ("src.clinical_guideline_agent.schema", "FileCitation"),
    ("src.drugs_agent.schema", "DrugAnswer"),
    ("src.drugs_agent.schema", "DrugMatch"),
]


class ClinicalDrugOrchestrator:
    """Sends a question through the guideline agent and, if needed, the drug agent."""

    def __init__(
        self,
        guideline_agent: Any,
        drug_agent: Any,
        chat_completions_provider: Any,
        *,
        chat_completion_model_name: str | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        # Use the model name we were given, otherwise fall back to the one in settings.
        model_name = chat_completion_model_name or get_settings().openai_model_name

        # The checkpointer is where conversations are remembered between questions.
        # The in-memory default forgets everything when the program stops; pass a
        # database-backed checkpointer to keep conversations across restarts.
        # Build the flowchart once, when the object is created.
        self.graph = build_orchestrator_graph(
            guideline_agent,
            drug_agent,
            chat_completions_provider,
            model_name,
            checkpointer
            if checkpointer is not None
            else InMemorySaver(
                serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_SAVED_CLASSES)
            ),
        )

    async def stream_clinical_drug_answer_updates(
        self, clinical_question: str, conversation_thread_id: str | None = None
    ) -> AsyncIterator[OrchestratorUpdate]:
        """Yield text pieces as they are written, then one final update with everything.

        Questions that share a `conversation_thread_id` share a history, so a later
        question can refer back to earlier ones. Without an id the question is
        answered on its own, as a brand-new one-question conversation.
        """
        # Step 1: reject empty questions.
        if not clinical_question.strip():
            raise ValueError("A question is required")

        # Step 2: tell LangGraph which conversation this question belongs to. The
        # thread id is the key under which the checkpointer saves and loads history.
        graph_run_configuration = {
            "configurable": {"thread_id": conversation_thread_id or str(uuid4())}
        }

        # Step 3: run the flowchart and forward everything it streams out.
        streamed_text_pieces: list[str] = []  # we keep these to rebuild the full answer later
        final_state: OrchestratorState = {}  # the graph's data once it has finished

        async for mode, payload in self.graph.astream(
            {
                # The `add_messages` reducer APPENDS this to the history that the
                # checkpointer loaded for this thread, so the history now ends with
                # the current question.
                "conversation_messages": [HumanMessage(content=clinical_question)],
                "clinical_question": clinical_question,
                # Clear the leftovers of the previous question on this thread (the
                # checkpointer saved them), so they cannot leak into this answer.
                "guideline_answer": None,
                "cited_guideline_text": "",
                "guideline_sources": [],
                "drug_classes": [],
                "drug_answer": None,
            },
            graph_run_configuration,
            stream_mode=["custom", "values"],
        ):
            if mode == "custom":
                # "custom" = a piece of answer text. Pass it straight to the caller.
                streamed_text_pieces.append(payload)
                yield OrchestratorUpdate(type="text", text=payload)
            else:
                # "values" = the graph's current data. The last one we see is the final one.
                final_state = payload

        # Step 4: read the final results out of the graph's data. There is no
        # guideline answer on the drug-only route, which never asks the guidelines.
        route = final_state["route"]
        guideline_answer = final_state.get("guideline_answer")

        # Step 5: work out the final text and the list of sources (citations).
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

        # Step 6: remember the answer. Nodes only stream text out, so the finished
        # answer (with its citation markers fixed) is only known here. We save it into
        # the same thread; the `add_messages` reducer appends it right after the
        # question, which keeps the history in question, answer, question, answer order.
        await self.graph.aupdate_state(
            graph_run_configuration,
            {"conversation_messages": [AIMessage(content=final_text)]},
        )

        # Step 7: send the final "completed" update with everything bundled together.
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
