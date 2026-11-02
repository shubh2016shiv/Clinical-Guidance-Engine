"""Answers medical questions using clinical guidelines and, when needed, drug data.

A user asks a question. The flowchart in `graph.py` decides which agents it needs
and runs them. This class runs that flowchart, passes the streamed text on to the
caller, and bundles the final answer together with its sources.
"""

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import HumanMessage

from src.config import get_settings
from src.memory import create_conversation_run_config, create_in_memory_checkpointer

from .graph import build_orchestrator_graph
from .schema import (
    OrchestratorAnswer,
    OrchestratorContext,
    OrchestratorState,
    OrchestratorUpdate,
)


# ---------------------------------------------------------------------------
# PREPARE QUESTION — Reset working results and supply the new user message
# ---------------------------------------------------------------------------
# Other state fields use replacement. add_messages merges by message ID: new IDs
# append, matching IDs update. Supplying one new message keeps earlier history.
# Official reference: https://docs.langchain.com/oss/python/langgraph/graph-api#working-with-messages-in-graph-state
def create_question_state(clinical_question: str) -> OrchestratorState:
    """Return fresh per-question fields and the user message to append to history.

    History is appended by the graph reducer, not replaced. Triage sets the route;
    all other per-question results are cleared so saved results cannot leak into it.
    """
    return {
        "conversation_messages": [HumanMessage(content=clinical_question)],
        "clinical_question": clinical_question,
        "drug_search_terms": [],
        "guideline_answer": None,
        "catalog_section_text": "",
        "final_text": "",
        "sources": [],
    }


# ---------------------------------------------------------------------------
# PUBLIC ENTRY — Own the lifecycle; graph.py chooses the route
# ---------------------------------------------------------------------------
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
        model_name = chat_completion_model_name or get_settings().openai_model_name

        # ---------------------------------------------------------------------------
        # SETUP 1 — Pass dependencies without saving clients in graph state
        # ---------------------------------------------------------------------------
        # Nodes read this through runtime.context during each run.
        # Official reference: https://docs.langchain.com/oss/python/langgraph/graph-api#runtime-context
        self.graph_context = OrchestratorContext(
            chat_completions_provider=chat_completions_provider,
            model_name=model_name,
            guideline_agent=guideline_agent,
            drug_agent=drug_agent,
        )

        # Keep a supplied checkpointer; otherwise remember conversations in memory.
        # ---------------------------------------------------------------------------
        # SETUP 2 — Retain a supplied saver or use process-local memory
        # ---------------------------------------------------------------------------
        # InMemorySaver forgets checkpoints when the process stops.
        # Official reference: https://docs.langchain.com/oss/python/langgraph/persistence
        if checkpointer is None:
            checkpointer = create_in_memory_checkpointer()
        self.graph = build_orchestrator_graph(checkpointer)

    async def stream_clinical_drug_answer_updates(
        self, clinical_question: str, conversation_thread_id: str | None = None
    ) -> AsyncIterator[OrchestratorUpdate]:
        """Yield text pieces as they are written, then one final update with everything.

        Questions that share a `conversation_thread_id` share a history, so a later
        question can refer back to earlier ones. Without an id, graph history starts
        fresh. A reused guideline agent still retains its own response chain
        independently of graph history.
        """
        # ---------------------------------------------------------------------------
        # VALIDATE QUESTION — Validate input and select the conversation
        # ---------------------------------------------------------------------------
        # configurable.thread_id selects graph history. It does not select the guideline
        # agent response chain, which belongs to that agent instance.
        # Official reference: https://docs.langchain.com/oss/python/langgraph/checkpointers#threads
        if not clinical_question.strip():
            raise ValueError("A question is required")

        conversation_run_config = create_conversation_run_config(conversation_thread_id)
        question_state = create_question_state(clinical_question)

        # ---------------------------------------------------------------------------
        # RUN NODES 01–04 — Forward custom text and progress events while the graph runs
        # ---------------------------------------------------------------------------
        # This code uses one stream mode and the default v1 format: our raw kind/text dict.
        # Current docs also show v2 envelopes. Do not change formats just to match an example.
        # Official reference: https://docs.langchain.com/oss/python/langgraph/streaming
        async for stream_event in self.graph.astream(
            question_state,
            conversation_run_config,
            context=self.graph_context,
            stream_mode="custom",
        ):
            yield OrchestratorUpdate(type=stream_event["kind"], text=stream_event["text"])

        # ---------------------------------------------------------------------------
        # DELIVER ANSWER — Read the completed snapshot and send one final update
        # ---------------------------------------------------------------------------
        # Use the same conversation configuration. Snapshot values contain the completed
        # text and sources saved by node 04, not merely the raw streamed chunks.
        # Official reference: https://docs.langchain.com/oss/python/langgraph/checkpointers#get-state
        final_state_snapshot = await self.graph.aget_state(conversation_run_config)
        final_state: OrchestratorState = final_state_snapshot.values

        yield OrchestratorUpdate(
            type="completed",
            answer=OrchestratorAnswer(
                route=final_state["route"],
                text=final_state["final_text"],
                sources=final_state["sources"],
                guideline_answer=final_state["guideline_answer"],
            ),
        )
