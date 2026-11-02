"""Classify a question from its text and selected conversation history."""

from typing import Any

from src.adapters.llm_provider.structured_model_response import ask_model_for_structured_json
from src.prompts.orchestrator_system_prompts import QUESTION_TRIAGE_SYSTEM_PROMPT

from .schema import QuestionTriage


# ---------------------------------------------------------------------------
# NODE 01 DETAIL — Ask for a route using selected earlier messages
# ---------------------------------------------------------------------------
# This ordinary model call returns raw search terms. The node adapter selects history,
# normalizes terms and chooses the route; no graph runtime is needed here.
async def classify_question(
    *,
    question: str,
    previous_messages: list[dict[str, str]],
    chat_completions_provider: Any,
    model_name: str,
) -> QuestionTriage:
    """Classify a question using already-selected previous messages, without graph state.

    Previous messages exclude the current question. Search terms are returned as the
    model supplied them; the graph adapter applies trimming and fallback rules.
    """
    return await ask_model_for_structured_json(
        chat_completions_provider,
        model_name,
        QUESTION_TRIAGE_SYSTEM_PROMPT,
        question,
        QuestionTriage,
        previous_conversation_messages=previous_messages,
    )
