"""Node: decide which agents the question needs."""

from typing import Any

from src.prompts.orchestrator_system_prompts import QUESTION_ROUTING_SYSTEM_PROMPT

from ..schema import OrchestratorState, QuestionRoutingDecision
from ..structured_model_call import ask_model_for_structured_json


def make_decide_route_node(chat_completions_provider: Any, model_name: str):
    """Build the node. The graph calls the returned function with its shared state."""

    async def decide_route(state: OrchestratorState) -> dict[str, Any]:
        """Ask the model: guidelines only, guidelines + drugs, or drugs only?"""
        decision = await ask_model_for_structured_json(
            chat_completions_provider,
            model_name,
            QUESTION_ROUTING_SYSTEM_PROMPT,
            state["clinical_question"],
            QuestionRoutingDecision,
        )
        return {"route": decision.route}

    return decide_route
