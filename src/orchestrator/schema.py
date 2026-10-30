"""Structured outputs, graph state and stream updates for the clinical-drug orchestrator."""

from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict

from src.clinical_guideline_agent import GuidelineAnswer
from src.clinical_guideline_agent.schema import FileCitation
from src.drugs_agent import DrugAnswer

QuestionRoute = Literal["clinical", "clinical_and_drug", "drug"]
"""Which agents answer a question: guidelines alone, both, or the drug catalog alone."""


class QuestionRoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: QuestionRoute


class RecommendedDrugClasses(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drug_classes: list[str]


class OrchestratorState(TypedDict, total=False):
    # Every user question and final assistant answer of this conversation, oldest first.
    # `add_messages` is a "reducer": when a node or caller returns new messages for this
    # key, LangGraph APPENDS them instead of replacing the list. The checkpointer saves
    # it between questions. See conversation_history.py for how it is read back.
    conversation_messages: Annotated[list[AnyMessage], add_messages]

    # Everything below belongs to ONE question. Because the checkpointer also saves
    # these, the orchestrator resets them at the start of every question so that an
    # earlier question's guideline or drug results can never leak into the next one.
    clinical_question: str
    route: QuestionRoute
    guideline_answer: GuidelineAnswer | None
    cited_guideline_text: str
    guideline_sources: list[FileCitation]
    drug_classes: list[str]
    drug_answer: DrugAnswer | None


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
    drug_classes: list[str]
    drug_answer: DrugAnswer | None


@dataclass(frozen=True)
class OrchestratorUpdate:
    type: Literal["text", "completed"]
    text: str = ""
    answer: OrchestratorAnswer | None = None
