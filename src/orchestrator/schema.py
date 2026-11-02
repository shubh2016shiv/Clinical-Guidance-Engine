"""Structured outputs, graph state and stream updates for the clinical-drug orchestrator."""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from src.clinical_guideline_agent import GuidelineAnswer
from src.clinical_guideline_agent.schema import FileCitation
from src.memory import ConversationHistoryState

# ---------------------------------------------------------------------------
# DATA 1 — Name our three application outcomes
# ---------------------------------------------------------------------------
# These are application route values, not LangGraph streaming modes.
QuestionRoute = Literal["clinical", "clinical_and_drug", "drug"]
"""What answered a question: the guidelines alone, the guidelines plus the catalog, or the
catalog alone. Triage selects the route from the question, not guideline class extraction."""


# ---------------------------------------------------------------------------
# DATA 2 — Validate the model decisions before using them
# ---------------------------------------------------------------------------
# QuestionTriage decides the initial workflow path.
class QuestionTriage(BaseModel):
    """Choose guidelines, drug catalog or both, and supply resolved drug search terms."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["catalog_only", "clinical", "clinical_and_drug"]
    # Queries for stock requests, resolved against history without class identification.
    drug_search_terms: list[str]


# ---------------------------------------------------------------------------
# DATA 3 — Declare checkpointed history and current working results
# ---------------------------------------------------------------------------
# Nodes return only changed keys. Ordinary keys use replacement; message history
# has a reducer. total=False is Python typing, not runtime data validation.
# Official reference: https://docs.langchain.com/oss/python/langgraph/graph-api#reducers
class OrchestratorState(ConversationHistoryState, total=False):
    """Checkpointed conversation history and the current question's working results.

    create_question_state() in orchestrator.py resets per-question results and supplies
    the new user message. Triage always sets the route before the graph branches.
    Fields remain optional because nodes return partial updates during execution.
    """

    # ConversationHistoryState supplies the history field and message reducer.
    # Their ownership is src/memory; only workflow-specific fields are declared here.

    # Per-question fields: reset before each run; triage sets the route.
    clinical_question: str
    route: QuestionRoute
    # Set by triage for stock requests; always empty on the clinical-only route.
    drug_search_terms: list[str]
    # None on the catalog-only route, which never asks the guidelines.
    guideline_answer: GuidelineAnswer | None
    # The drug agent's text exactly as `search_catalog` forwarded it; empty when the
    # catalog was never searched.
    catalog_section_text: str
    # Set by the last node, `finalize_answer`: the answer the user keeps, with its
    # citation markers in place, and the guideline files those markers number.
    final_text: str
    sources: list[FileCitation]


# ---------------------------------------------------------------------------
# DATA 4 — Keep supplied clients separate from checkpointed state
# ---------------------------------------------------------------------------
# runtime.context carries these dependencies during a run.
# Official reference: https://docs.langchain.com/oss/python/langgraph/graph-api#runtime-context
@dataclass(frozen=True)
class OrchestratorContext:
    """Clients and model selection supplied to graph adapters, never checkpointed."""

    chat_completions_provider: Any
    model_name: str
    guideline_agent: Any
    drug_agent: Any


# ---------------------------------------------------------------------------
# DATA 5 — Bundle the final text with sources and agent results
# ---------------------------------------------------------------------------
# This is what callers receive after successful finalization.
@dataclass(frozen=True)
class OrchestratorAnswer:
    """Everything one question produced, for callers that cite or save it.

    ``text`` carries inline markers such as ``${}^{[1]}$``; ``sources`` lists the files they number.
    ``guideline_answer`` is None on the ``drug`` route, which never asks the guidelines.
    """

    route: QuestionRoute
    text: str
    sources: list[FileCitation]
    guideline_answer: GuidelineAnswer | None


# ---------------------------------------------------------------------------
# DATA 6 — Distinguish text, progress and completion
# ---------------------------------------------------------------------------
# This is our public event type, not a LangGraph SDK streaming envelope.
@dataclass(frozen=True)
class OrchestratorUpdate:
    """One thing the orchestrator sends out while answering.

    ``text`` is a piece of the answer, to be added to what is already shown.
    ``status`` says what is being worked on and replaces the last status shown;
    it is not part of the answer. ``completed`` comes last and carries ``answer``.
    """

    type: Literal["text", "status", "completed"]
    text: str = ""
    answer: OrchestratorAnswer | None = None
